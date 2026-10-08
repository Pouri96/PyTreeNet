"""Checks of rule/spcfast.py: the F-form objective against the window-marginal objective, the adjoint pair, the linearisation."""
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast

rng = np.random.default_rng(1)
model, N, chi = "ising", 8, 3
import os as _os
cut = spcfast.SPCFast(model, N, a=2, taus=[1.0, 2.0], fw=0.0, two=bool(int(_os.environ.get('TWO', '0'))))
cut.debug = []
M.run_tebd(model, N, chi, 25, 0.1, cut, gates=M.make_gates(model, N, 0.1))
print('cuts', cut.calls, 'fired', cut.fired, 'debug', len(cut.debug))

# 1. F-form against the window-marginal objective on random density matrices (region L=6, bulk)
L, ks = 6, (1, 2, 3)
D, (Fs, Fd32), iu, dg = cut._region(L, False)
F = np.concatenate([Fs.toarray(), Fd32.astype(float)], axis=0)
def rand_rho(n):
    W = rng.normal(size=(D, n)) + 1j * rng.normal(size=(D, n))
    r = W @ W.conj().T
    return r / np.trace(r).real
def hvec(rho):
    return np.concatenate([rho[dg, dg].real, rho[iu].real, rho[iu].imag])
Hm = np.zeros((D, D), dtype=complex)
for bb in range(L - 1):
    Hm += np.kron(np.kron(np.eye(2 ** bb), M.h_bond(model, 3, N)), np.eye(2 ** (L - bb - 2)))
ev, V = np.linalg.eigh(Hm)
def direct(rho, tgt):
    tot = 0.0
    for tau in cut.taus:
        U = (V * np.exp(-1j * tau * ev)) @ V.conj().T
        d = U @ (rho - tgt) @ U.conj().T
        for k in ks:
            nwin = L - k + 1
            acc = 0.0
            for off in range(nwin):
                r = L - off - k
                X = d.reshape(2 ** off, 2 ** k, 2 ** r, 2 ** off, 2 ** k, 2 ** r)
                acc += np.sum(np.abs(np.einsum('asbatb->st', X)) ** 2)
            tot += acc / nwin
    return tot
a, b = rand_rho(20), rand_rho(30)
rv = F @ (hvec(a) - hvec(b))
print('objective F-form vs direct', rv @ rv, direct(a, b))

# 2. adjoint pair and linearisation at every recorded cut
worst_adj, worst_lin = 0.0, 0.0
for d in cut.debug:
    x = rng.normal(size=d['nx'])
    g = rng.normal(size=d['nres'])
    lhs, rhs = g @ d['jvp'](x), d['vjp'](g) @ x
    worst_adj = max(worst_adj, abs(lhs - rhs) / (abs(lhs) + 1e-30))
    eps = 1e-6 / np.linalg.norm(x)
    r_plus = d['exact'](eps * x)[2]
    num = (r_plus - d['r0']) / eps
    ana = d['jvp'](x)
    worst_lin = max(worst_lin, np.linalg.norm(num - ana) / np.linalg.norm(ana))
print('adjoint rel err (max)', worst_adj, ' linearisation rel err (max)', worst_lin)

# 3. the matmul form of theta -> W against the tensordot form
dd = cut.debug[len(cut.debug) // 2]
th = rng.normal(size=(dd['l'], 2, 2, dd['r'])) + 1j * rng.normal(size=(dd['l'], 2, 2, dd['r']))
print('W form difference', np.abs(dd['Wof'](th) - dd['Wold'](th)).max())

# 4. linearisation error against the step size (second-order behaviour): should shrink linearly with the step until float32 noise
for rel in (1e-1, 1e-2, 1e-3, 1e-4):
    errs = []
    for d in cut.debug[::5]:
        x = rng.normal(size=d['nx'])
        x *= rel / np.linalg.norm(x)
        num = d['exact'](x)[2] - d['r0']
        ana = d['jvp'](x)
        errs.append(np.linalg.norm(num - ana) / np.linalg.norm(ana))
    print('step', rel, 'median linearisation rel err', np.median(errs), 'max', np.max(errs))
