"""MPS-native lookahead marginal fit.

The fit of mfc.py is extended by the expectation values of Heisenberg-evolved Pauli strings,
    <w| (U^dag)^m P U^m |w>,   P a 1- or 2-site Pauli string, m in a short list of Trotter steps,
which equal the marginals of U^m w without ever forming U^m w. The operators do not depend on the state, so their
MPOs (heis_ops.py, truncated at relative tolerance eps) are built once. Each MPO is cut to the window outside which
every tensor is a scalar multiple of the identity, and all windows of one m are padded to a common shape so that the
expectation values of the whole bank are one vmapped scan over the window sites.
"""
import os
import sys
if os.environ.get('PYLIBS'):
    sys.path.insert(0, os.environ['PYLIBS'])
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)
import mpsenh as M
import heis_ops as H
import mfc

TRIVIAL_TOL = 1e-7


def _trivial_scalar(t):
    """(scalar s, True) if t has bond dimension 1 on both sides and equals s * identity."""
    if t.shape[0] != 1 or t.shape[3] != 1:
        return None
    m = t[0, :, :, 0]
    s = np.trace(m) / 2
    if abs(s) > 0 and np.linalg.norm(m - s * np.eye(2)) <= TRIVIAL_TOL * abs(s):
        return s
    return None


class Bank:
    """Heisenberg-evolved operator MPOs for one m, as padded arrays.

    start  (n,)             first chain site of the window of each operator
    Wts    list over window sites of (n, 2*Dr_j, Dl_j*2) MPO tensors as matrices [(o,v),(w,i)], zero padded to the largest bond
    scal   (n,)             product of the scalars of the trivial tensors outside the window
    kk     (n,)             size of the Pauli string the operator started from
    """
    def __init__(self, m, N, start, Wts, scal, kk, W, Dp, lens=None):
        self.m, self.N, self.start, self.Wts, self.scal, self.kk, self.W, self.Dp = m, N, start, Wts, scal, kk, W, Dp
        self.lens = lens

    @property
    def n(self):
        return len(self.start)


def build_banks(model, N, dt, ms, ks=(1, 2), eps=1e-5, dmax=150, coarse=0.0, gamma=0.0):
    """dict m -> Bank, for every Pauli string of size in ks (adjacent windows) and every m in ms.

    coarse = 0: m counts Strang sweep steps of the dynamics (dt). coarse > 0: m counts brickwork Strang steps of time
    `coarse`, a propagator of strict light cone chosen only to define the lookahead operators.
    """
    Gs = M.make_gates(model, N, dt)
    stepper = H.brick_stepper(M.make_gates(model, N, coarse), M.make_gates(model, N, 2 * coarse)) if coarse > 0 else None
    strings = [s for s in H.operator_set(N, kmax=max(ks)) if len(s[0]) in ks]
    mpos = {m: [] for m in ms}
    for sites, labels in strings:
        res = H.evolve(H.local_op(N, sites, labels), Gs, N, max(ms), list(ms), eps=eps, dmax=dmax, stepper=stepper, gamma=gamma)
        for m in ms:
            mpos[m].append(res[m])
    banks = {}
    for m in ms:
        supports, scalars = [], []
        for W in mpos[m]:
            sc = [_trivial_scalar(t) for t in W]
            idx = [i for i, s in enumerate(sc) if s is None]
            supports.append((idx[0], idx[-1]) if idx else (0, 0))
            scalars.append(sc)
        Wmax = max(h - l + 1 for l, h in supports)
        n = len(strings)
        starts = np.zeros(n, dtype=np.int64)
        scal = np.ones(n, dtype=complex)
        kk = np.array([len(s[0]) for s in strings])
        place = []
        for o, (W, (lo, hi), sc) in enumerate(zip(mpos[m], supports, scalars)):
            s0 = min(lo, N - Wmax)
            starts[o] = s0
            tens = []
            for j in range(N):
                if s0 <= j < s0 + Wmax:
                    tens.append(W[j])
                else:
                    assert sc[j] is not None, f'operator support leaks outside the window (m={m}, string {strings[o]}, site {j})'
                    scal[o] *= sc[j]
            place.append(tens)
        Dl = [max(place[o][j].shape[0] for o in range(n)) for j in range(Wmax)]
        Dr = [max(place[o][j].shape[3] for o in range(n)) for j in range(Wmax)]
        Wts = []
        for j in range(Wmax):
            arr = np.zeros((n, Dl[j], 2, 2, Dr[j]), dtype=complex)           # (w, o, i, v)
            for o in range(n):
                t = place[o][j]
                arr[o, :t.shape[0], :, :, :t.shape[3]] = t
            Wts.append(np.ascontiguousarray(arr.transpose(0, 2, 4, 1, 3)).reshape(n, 2 * Dr[j], Dl[j] * 2))   # [(o,v),(w,i)]
        Dp = max(max(Dl), max(Dr))
        banks[m] = Bank(m, N, starts, Wts, scal, kk, Wmax, Dp, np.array([h - l + 1 for l, h in supports]))
    return banks


def _expect_one(Twin, E0, Rend, Wts):
    """<w|O|w> for one operator. Wts[j] (2*Dr_j, Dl_j*2) holds MPO tensor j as the matrix [(o, v), (w, i)].

    Every step is a plain matrix product on reshaped arrays, no transposes:
      X[(a,w), (i,d)]   = E[(a,w), b] A[b, (i,d)]
      Y[a, (o,v), d]    = Wts[(o,v), (w,i)] X[a, (w,i), d]
      E'[c, (v,d)]      = conj(A)[(a,o), c]^T Y[(a,o), (v,d)]
    """
    c = E0.shape[0]
    E = E0[:, None, :]
    for j, Wj in enumerate(Wts):
        A = Twin[j]
        Dl, Dr = Wj.shape[1] // 2, Wj.shape[0] // 2
        X = (E.reshape(c * Dl, c) @ A.reshape(c, 2 * c)).reshape(c, Dl * 2, c)
        Y = jnp.matmul(Wj[None], X)
        E = (jnp.conj(A).reshape(c * 2, c).T @ Y.reshape(c * 2, Dr * c)).reshape(c, Dr, c)
    return jnp.einsum('cd,cd->', E[:, 0, :], Rend)


def _pad_sq(X, c):
    return jnp.pad(X, ((0, c - X.shape[0]), (0, c - X.shape[1])))


def bank_values(Ts, bank_arrays, W, batch, single=True):
    """Normalised real expectation values <w|O_o|w> for all operators of one bank, Ts a list of (l,2,r) arrays.

    The contraction runs in complex64 when `single` (the values enter the objective at the 1e-4 level)."""
    N = len(Ts)
    c = max(t.shape[2] for t in Ts)
    L, R = mfc._env_left(Ts), mfc._env_right(Ts)
    nrm = jnp.real(L[N][0, 0])
    cd = jnp.complex64 if single else jnp.complex128
    Tstack = jnp.stack([jnp.pad(t, ((0, c - t.shape[0]), (0, 0), (0, c - t.shape[2]))) for t in Ts]).astype(cd)
    Lst = jnp.stack([_pad_sq(x, c) for x in L]).astype(cd)
    Rst = jnp.stack([_pad_sq(x, c) for x in R]).astype(cd)
    start, Wts, scal = bank_arrays

    def one(args):
        s, Wo, sc = args
        Twin = jax.lax.dynamic_slice_in_dim(Tstack, s, W, axis=0)
        return jnp.real(sc * _expect_one(Twin, Lst[s], Rst[s + W], [w.astype(cd) for w in Wo]).astype(jnp.complex128))
    return jax.lax.map(one, (start, list(Wts), scal), batch_size=batch) / nrm


def bank_args(bank):
    return (jnp.asarray(bank.start), [jnp.asarray(w) for w in bank.Wts], jnp.asarray(bank.scal))


def make_objective(shapes, ks, fw, weights, banks, lam, batch, region=None, lam_r=1.0):
    ms = sorted(banks)
    # weight of an operator: Frobenius norm of a k-site Pauli expansion is (1/2^k) sum_P |<P>|^2, averaged over windows
    opw = {}
    for m in ms:
        kk = banks[m].kk
        w = np.zeros(banks[m].n)
        for k in set(kk.tolist()):
            w[kk == k] = 1.0 / (2 ** k * (banks[m].N - k + 1))
        opw[m] = jnp.asarray(w)
    Wsz = {m: banks[m].W for m in ms}

    def unpack(x):
        out, p = [], 0
        for sh in shapes:
            n = int(np.prod(sh))
            out.append((x[p:p + n] + 1j * x[p + n:p + 2 * n]).reshape(sh))
            p += 2 * n
        return out

    def f(x, taus, tgt, bank_arr, Tw, rtgt):
        Ts = unpack(x)
        marg = mfc.marginals(Ts, ks)
        tot = 0.0
        for k in ks:
            tot = tot + weights[k] * jnp.mean(jnp.sum(jnp.abs(marg[k] - taus[k]) ** 2, axis=(1, 2)))
        for m in ms:
            v = bank_values(Ts, bank_arr[m], Wsz[m], batch)
            tot = tot + lam * jnp.sum(opw[m] * (v - tgt[m]) ** 2)
        if region is not None:
            tot = tot + lam_r * region.loss(region.values(Ts), rtgt)
        if fw > 0:
            nT = jnp.real(mfc.overlap(Ts, Ts))
            nW = jnp.real(mfc.overlap(Tw, Tw))
            tot = tot + fw * (1.0 - jnp.abs(mfc.overlap(Tw, Ts)) ** 2 / (nT * nW))
        return tot
    return jax.jit(jax.value_and_grad(f)), unpack


class LookaheadFit:
    def __init__(self, banks, ks=(1, 2, 3), fw=0.03, maxiter=250, lam=1.0, weights=None, batch=60, region=None, lam_r=1.0):
        self.banks, self.ks, self.fw, self.maxiter, self.lam, self.batch = banks, tuple(ks), fw, maxiter, lam, batch
        self.region, self.lam_r = region, lam_r
        self._rt = jax.jit(region.values) if region is not None else None
        self.weights = weights or {k: 1.0 for k in ks}
        self.bank_arr = {m: bank_args(b) for m, b in banks.items()}
        self._cache = {}
        self._tgt = {}
        self.evals = 0
        self.last = None

    def targets(self, Tw):
        key = tuple(t.shape for t in Tw)
        if key not in self._tgt:
            Wsz = {m: b.W for m, b in self.banks.items()}
            self._tgt[key] = jax.jit(lambda Ts, arr: {m: bank_values(Ts, arr[m], Wsz[m], self.batch) for m in self.banks})
        return self._tgt[key]([jnp.asarray(t) for t in Tw], self.bank_arr)

    def __call__(self, Tw, chi, init):
        shapes = tuple(t.shape for t in init)
        wshapes = tuple(t.shape for t in Tw)
        key = (shapes, wshapes)
        if key not in self._cache:
            self._cache[key] = make_objective(list(shapes), self.ks, self.fw, self.weights, self.banks, self.lam, self.batch,
                                              region=self.region, lam_r=self.lam_r)
        vg, unpack = self._cache[key]
        Twj = [jnp.asarray(t) for t in Tw]
        taus = {k: v for k, v in mfc.marginals(Twj, self.ks).items()}
        tgt = self.targets(Tw)
        rtgt = self._rt(Twj) if self._rt is not None else {}

        def fun(x):
            val, g = vg(jnp.asarray(x), taus, tgt, self.bank_arr, Twj, rtgt)
            self.evals += 1
            return float(val), np.asarray(g)
        x0 = mfc.pack(init)
        f0 = fun(x0)[0]
        res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=self.maxiter, maxfun=2 * self.maxiter, ftol=1e-15, gtol=1e-11))
        self.last = (f0, float(res.fun), int(res.nit))
        return [np.asarray(t) for t in unpack(res.x)]


def run_mfcl(model, N, chi, nsteps, dt, chi_w_factor=2, ks=(1, 2, 3), fw=0.03, maxiter=250, ms=(3, 6), kla=(1, 2),
             eps=1e-5, lam=1.0, batch=60, progress=None, banks=None, coarse=0.0, gamma=0.0, dmax=150, region=None, lam_r=1.0):
    """Like mfc.run_mfc(arm='mfc') with the lookahead terms of the operator banks added to the fit."""
    Gs = M.make_gates(model, N, dt)
    T = mfc.right_canonicalize(M.initial_mps(model, N))
    chi_w = int(round(chi_w_factor * chi))
    banks = banks if banks is not None else (build_banks(model, N, dt, ms, kla, eps, dmax=dmax, coarse=coarse, gamma=gamma) if ms else {})
    fc = LookaheadFit(banks, ks=ks, fw=fw, maxiter=maxiter, lam=lam, batch=batch, region=region, lam_r=lam_r)
    peak, nfit, hist = 1, 0, []
    for step in range(nsteps):
        T = mfc.sweep_step(T, Gs, chi_w)
        T = mfc.normalise(mfc.right_canonicalize(T))
        peak = max(peak, mfc.max_bond(T))
        if mfc.max_bond(T) > chi:
            T0 = mfc.normalise(mfc.compress_svd(T, chi))
            init = mfc.pad_to(T0, chi)
            Tw = mfc.pad_to(T, chi_w)
            T = fc(Tw, chi, init)
            hist.append(fc.last)
            nfit += 1
            T = mfc.normalise(mfc.right_canonicalize(T))
        if progress:
            progress(step, T)
    return T, dict(peak=peak, nfit=nfit, hist=hist, chi_w=chi_w, evals=fc.evals)
