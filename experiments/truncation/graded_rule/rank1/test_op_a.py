"""Correctness checks, part A (no tilt): gates, dense Pauli-space circuit against explicit matrices, scorers, MPS against dense at chi = inf,
reweighted frame, the 'svdraw is svd times a scalar' identity, and the TFIM null control."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import op_tebd as O
import mpsenh as M
import pauli_prop as PP

ok_all = True


def check(name, val, tol):
    global ok_all
    ok = bool(val <= tol)
    ok_all &= ok
    print(f'{"PASS" if ok else "FAIL"}  {name}: {val:.3e} (tol {tol:.0e})', flush=True)


# 1. gates: h_bond equals the repository's, R orthogonal, gate array reshaping
for model in ('ising', 'heis'):
    e = max(np.abs(O.h_bond(model, b, 6) - M.h_bond(model, b, 6)).max() for b in range(5))
    check(f'h_bond {model} == mpsenh.h_bond', e, 1e-15)
Gs = O.make_gates4('ising', 6, 0.1)
Gm = M.make_gates('ising', 6, 0.1)
check('gates == mpsenh.make_gates', max(np.abs(a - m.reshape(4, 4)).max() for a, m in zip(Gs, Gm)), 1e-15)
R = PP.transfer(Gs[2])
check('R orthogonal', np.abs(R @ R.T - np.eye(16)).max(), 1e-13)

# 2. dense Pauli-space circuit against explicit matrices O(t) = U^dag O U, U = G_n ... G_1 (G_1 acts first on states)
def embed(G4, b, N):
    return np.kron(np.kron(np.eye(2 ** b), G4), np.eye(2 ** (N - b - 2)))

def pauli_coeffs(Om, N):
    """c_P = Tr(P O)/2^N for all 4^N strings, C order (site 0 slowest)."""
    t = Om.reshape([2] * N + [2] * N)
    # reshape trick: coefficients via tensor contraction with single-site Pauli bases
    for i in range(N):
        pass
    c = np.zeros(4 ** N)
    import itertools
    for idx, labs in enumerate(itertools.product(range(4), repeat=N)):
        P = np.eye(1)
        for l_ in labs:
            P = np.kron(P, O.PL[l_])
        c[idx] = np.trace(P @ Om).real / 2 ** N
    return c

for model in ('ising', 'heis'):
    N, nsteps, dt = 5, 3, 0.2
    Gs = O.make_gates4(model, N, dt)
    seq = PP.circuit_bonds(N, nsteps)
    U = np.eye(2 ** N, dtype=complex)
    for b in seq:
        U = embed(Gs[b], b, N) @ U
    Z0 = np.kron(M.Z, np.eye(2 ** (N - 1)))
    Ot = U.conj().T @ Z0 @ U
    ce = pauli_coeffs(Ot, N)
    gops = O.pauli_gates(model, N, dt)
    G16 = [g.reshape(16, 16) for g in gops]
    c = O.init_dense(N)
    for b in seq[::-1]:
        c = O.dense_gate(c, G16[b], b, N)
    check(f'dense Pauli circuit == U^dag Z0 U ({model}, N=5, 3 steps)', np.abs(c - ce).max(), 1e-12)
    check(f'norm sum c^2 = 1 ({model})', abs(c @ c - 1), 1e-12)
    # OTOC formula: C_Z(x) = (1/2) Tr([Z_x, O]^dag [Z_x, O]) / 2^N = 2 (p_x(X) + p_x(Y))
    p1 = O.marginal1(c, N)
    worst = 0.0
    for x in range(N):
        Zx = np.kron(np.kron(np.eye(2 ** x), M.Z), np.eye(2 ** (N - x - 1)))
        Cm = Zx @ Ot - Ot @ Zx
        Cd = 0.5 * np.trace(Cm.conj().T @ Cm).real / 2 ** N
        worst = max(worst, abs(Cd - O.C_from_p1(p1[x], 'Z')))
        Xx = np.kron(np.kron(np.eye(2 ** x), M.X), np.eye(2 ** (N - x - 1)))
        Cm = Xx @ Ot - Ot @ Xx
        Cd = 0.5 * np.trace(Cm.conj().T @ Cm).real / 2 ** N
        worst = max(worst, abs(Cd - O.C_from_p1(p1[x], 'X')))
    check(f'OTOC C_Z, C_X from marginals vs commutator norm ({model})', worst, 1e-12)
    # <Neel| O(t) |Neel> via pauli_prop (independent code path, same circuit) vs the dense coefficient vector
    ex_pp = PP.propagate(model, N, nsteps, dt, {0: 3}, 0.0)[0]
    neel = M.mps_to_dense(M.neel_mps(N))
    ex_mat = np.vdot(neel, Ot @ neel).real
    check(f'pauli_prop.propagate == matrix expectation ({model})', abs(ex_pp - ex_mat), 1e-12)

# 3. MPS (chi = inf) against dense, N = 8, ising and heis and tfim, incl. the reweighted frame and the 'raw' cut
for model in ('ising', 'heis', 'tfim'):
    N, nsteps, dt = 8, 8, 0.1
    gops = O.pauli_gates(model, N, dt)
    p1d, cz0d, snaps = O.dense_series(model, N, nsteps, dt, snap_steps=(nsteps,))
    cd = snaps[nsteps]
    T, _ = O.run_op_tebd(N, 4 ** (N // 2), nsteps, gops, O.svd_cut_op)
    cm = O.mps_to_dense(T)
    check(f'MPS(chi=inf) == dense ({model}, N=8, 8 steps)', np.abs(cm - cd).max(), 1e-11)
    check(f'  max bond {max(O.ranks(T))}', 0.0, 1.0)
    g = 1.6
    gw = O.pauli_gates(model, N, dt, gamma=g)
    Tw, _ = O.run_op_tebd(N, 4 ** (N // 2), nsteps, gw, O.svd_cut_op)
    cw = O.phys_dense(Tw, g)
    check(f'rw:1.6 frame, MPS(chi=inf) back-transformed == dense ({model})', np.abs(cw - cd).max(), 1e-10)
    # truncated: svdraw (unnormalised cut) = svd * scalar
    chi = 6
    Ts, _ = O.run_op_tebd(N, chi, nsteps, gops, O.svd_cut_op)
    Tr, _ = O.run_op_tebd(N, chi, nsteps, gops, O.RawSVDCut())
    cs, cr = O.mps_to_dense(Ts), O.mps_to_dense(Tr)
    n = np.linalg.norm(cr)
    dif = np.abs(cr / n - cs / np.linalg.norm(cs)).max()
    if model == 'heis':
        # SU(2) symmetry: singular values at the rank boundary are exactly degenerate at some cuts (checked separately: 6 of 75 cuts
        # have a relative gap < 1e-10), so WHICH vectors of the multiplet are kept depends on rounding of the (scaled) input.
        print(f'INFO  raw cut vs svd*scalar, chi=6 (heis, degenerate boundary singular values): {dif:.3e}, norm={n:.4f}')
    else:
        check(f'raw cut == svd * scalar, chi=6 ({model}), norm={n:.4f}', dif, 1e-10)

# 4. TFIM (free fermion) null control: chi = 4 exact at N = 10, T = 6
N, T_, dt = 10, 6.0, 0.1
ns = int(round(T_ / dt))
gops = O.pauli_gates('tfim', N, dt)
p1d, cz0d, snaps = O.dense_series('tfim', N, ns, dt, snap_steps=(ns,))
for chi in (2, 4):
    T, _ = O.run_op_tebd(N, chi, ns, gops, O.svd_cut_op)
    cm = O.mps_to_dense(T)
    check(f'TFIM N=10 T=6 chi={chi}: max |c - c_ex|', np.abs(cm - snaps[ns]).max(), 1e-9)
print('ALL PASS' if ok_all else 'SOME FAILED')
