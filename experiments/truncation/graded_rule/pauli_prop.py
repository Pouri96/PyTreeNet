"""Pauli propagation baseline: Heisenberg-evolve a local observable through the SAME Trotter circuit as the MPS runs,
keep only Pauli strings with |coefficient| >= eps, and evaluate on the Neel product state. No state is stored.

Strings are packed 2 bits per site (I=0, X=1, Y=2, Z=3), site i at bits 2i. Each two-site gate maps a two-site Pauli
to at most 16 Paulis through a real 16x16 orthogonal matrix R, G^dag P G = sum_j R[P, j] P_j.
"""
import time
import numpy as np
import _paths  # noqa
import mpsenh as M

PL = [M.I2, M.X, M.Y, M.Z]


def transfer(G4):
    """R[c_in, c_out], code c = p_b + 4 p_{b+1}; G4 is the 4x4 gate on (b, b+1), kron(b, b+1) ordering."""
    R = np.zeros((16, 16))
    P2 = [np.kron(PL[c % 4], PL[c // 4]) for c in range(16)]     # site b is the low digit and the left kron factor
    for ci in range(16):
        Ev = G4.conj().T @ P2[ci] @ G4
        for co in range(16):
            R[ci, co] = np.trace(P2[co].conj().T @ Ev).real / 4
    return R


def circuit_bonds(N, nsteps):
    one = list(range(N - 1)) + list(range(N - 2, -1, -1))
    return one * nsteps


def propagate(model, N, nsteps, dt, start_string, eps, cap=4_000_000, tmax=1500):
    """start_string: dict {site: pauli code}. Returns (expectation on Neel, max strings, seconds, truncated flag)."""
    Gs = M.make_gates(model, N, dt)
    Rs = [transfer(G.reshape(4, 4)) for G in Gs]
    key0 = 0
    for s, p in start_string.items():
        key0 |= p << (2 * s)
    keys = np.array([key0], dtype=np.int64)
    coef = np.array([1.0])
    seq = circuit_bonds(N, nsteps)[::-1]
    t0, kmax, trunc = time.time(), 1, False
    for b in seq:
        R = Rs[b]
        sh = 2 * b
        code = (keys >> sh) & 15
        touched = code != 0
        if not touched.any():
            continue
        ku, cu = keys[~touched], coef[~touched]
        kt, ct, codet = keys[touched], coef[touched], code[touched]
        base = kt & ~np.int64(15 << sh)
        outk, outc = [ku], [cu]
        for j in range(16):
            r = R[codet, j]
            m = np.abs(r) > 1e-14
            if m.any():
                outk.append(base[m] | np.int64(j << sh))
                outc.append(ct[m] * r[m])
        nk, nc = np.concatenate(outk), np.concatenate(outc)
        uk, inv = np.unique(nk, return_inverse=True)
        uc = np.bincount(inv, weights=nc, minlength=len(uk))
        keep = np.abs(uc) >= eps
        keys, coef = uk[keep], uc[keep]
        kmax = max(kmax, len(keys))
        if len(keys) > cap or time.time() - t0 > tmax:
            trunc = True
            break
    # evaluate on the Neel product state: only I and Z contribute, Z = +1 on even sites (up), -1 on odd sites
    val = np.ones(len(keys))
    ok = np.ones(len(keys), dtype=bool)
    for s in range(N):
        d = (keys >> (2 * s)) & 3
        ok &= (d == 0) | (d == 3)
        val *= np.where(d == 3, 1.0 if s % 2 == 0 else -1.0, 1.0)
    return float(np.sum(coef * val * ok)), kmax, time.time() - t0, trunc


def sample_observables(N):
    i0 = N // 2 - 1
    obs = []
    for p in (1, 2, 3):
        obs.append(({i0: p}, ('single', i0, p)))
    for p in (1, 2, 3):
        for q in (1, 2, 3):
            obs.append(({i0: p, i0 + 1: q}, ('nn', i0, p, q)))
    for p in (1, 2, 3):
        for q in (1, 2, 3):
            obs.append(({i0: p, i0 + 2: q}, ('nnn', i0, p, q)))
    return obs


def exact_values(v, N, obs):
    v = v / np.linalg.norm(v)
    out = []
    for st, _ in obs:
        t = v.reshape([2] * N)
        w = t.copy()
        for s, p in st.items():
            w = np.moveaxis(np.tensordot(PL[p], np.moveaxis(w, s, 0), axes=([1], [0])), 0, s)
        out.append(float(np.vdot(t, w).real))
    return np.array(out)


if __name__ == '__main__':
    # correctness: N=6, 5 steps, eps=0 against the dense reference
    model, N, dt, ns = 'ising', 6, 0.1, 5
    ex = M.dense_reference(model, N, ns, dt)
    obs = sample_observables(N)
    exv = exact_values(ex, N, obs)
    got = np.array([propagate(model, N, ns, dt, st, 0.0)[0] for st, _ in obs])
    print('max |Pauli prop - dense| over', len(obs), 'observables:', np.max(np.abs(got - exv)))
