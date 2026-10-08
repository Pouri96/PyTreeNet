"""Rank 7, T3: the 1D kicked Ising circuit under TEBD with a cut rule.  New module, lives here so that rule/ stays untouched.

One Floquet step  U = [prod_b exp(-i thJ/2 Z_b Z_{b+1})] [prod_i exp(-i thh/2 X_i)]  (IBM: thJ = -pi/2, i.e. exp(+i pi/4 ZZ)) is applied as ONE
sweep of two-site gates, the direction alternating from step to step:

  right sweep (b = 0 ... N-2):  gate_b = ZZ_{b,b+1}(thJ) . RX_{b+1}(thh)         (RX_0 is folded into b = 0)
  left sweep  (b = N-2 ... 0):  gate_b = ZZ_{b,b+1}(thJ) . RX_b(thh)             (RX_{N-1} is folded into b = N-2)

Every site is kicked before both of the ZZ gates it takes part in, and the ZZ gates commute, so a sweep is exactly U.  The dense reference applies
the identical gate list with mpsenh.dense_gate.  Initial state |up>^N (Z = +1).  A cut is called as in mpsenh.run_tebd: svd_cut(theta, chi, dirn)
or SPCFast(theta, chi, dirn, A, B, b); SPCFast gets taus=() (static window targets only), because there is no Hamiltonian to look ahead with.

    python rank7/floquet.py A svd 32 out.npz [N=20] [depth=20]
    python rank7/floquet.py A spcf:2:0:4:none:1-2-3:0:1:1e-7:all:0.3 32 out.npz
    python rank7/floquet.py selftest
Settings: A = (thJ -pi/2, thh 0.7), B = (-pi/2, 1.0), C = (-pi/4, 0.7), D = (-pi/2, pi/2) Clifford control.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import _paths  # noqa: E402,F401
import mpsenh as M  # noqa: E402
import ladder_t as L  # noqa: E402

CACHE = HERE / '_refcache'
RES = HERE / 'results'
SETTINGS = {'A': (-np.pi / 2, 0.7), 'B': (-np.pi / 2, 1.0), 'C': (-np.pi / 4, 0.7), 'D': (-np.pi / 2, np.pi / 2)}
I2, X, Z = M.I2, M.X, M.Z


def rx(th):
    return np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * X


def zz(thJ):
    return np.diag(np.exp(-1j * thJ / 2 * np.diag(np.kron(Z, Z)).real))


def make_kicked_gates(N, thJ, thh, direction):
    """Gates (2,2,2,2) indexed by the bond b, for a right ('R') or left ('L') sweep, in the layout of mpsenh.make_gates."""
    Z2, K1 = zz(thJ), rx(thh)
    gates = []
    for b in range(N - 1):
        if direction == 'R':
            K = np.kron(K1, K1) if b == 0 else np.kron(I2, K1)
        else:
            K = np.kron(K1, K1) if b == N - 2 else np.kron(K1, I2)
        gates.append((Z2 @ K).reshape(2, 2, 2, 2))
    return gates


def all_up_mps(N):
    return [np.array([1, 0], dtype=complex).reshape(1, 2, 1) for _ in range(N)]


def sweep_order(N, direction):
    return range(N - 1) if direction == 'R' else range(N - 2, -1, -1)


def direction_of(step):
    return 'R' if step % 2 == 1 else 'L'


def ref_trajectory(label, N, depth):
    thJ, thh = SETTINGS[label]
    fs = CACHE / f'floq_{label}_N{N}_d{depth}.npy'
    fo = RES / f't3_ref_obs_{label}_N{N}_d{depth}.npz'
    if not (fs.exists() and fo.exists()):
        CACHE.mkdir(exist_ok=True)
        RES.mkdir(exist_ok=True)
        gates = {d: make_kicked_gates(N, thJ, thh, d) for d in 'RL'}
        v = M.mps_to_dense(all_up_mps(N))
        states = np.zeros((depth, 2 ** N), dtype=complex)
        obs = {k: [] for k in ('single', 'nn', 'nnn', 'z', 'zz')}
        for step in range(1, depth + 1):
            d = direction_of(step)
            for b in sweep_order(N, d):
                v = M.dense_gate(v, gates[d][b], b, N)
            states[step - 1] = v
            o = L.observables(v, N, 'ising')
            for k in obs:
                obs[k].append(o[k])
        np.save(fs, states)
        np.savez(fo, t=np.arange(1, depth + 1), **{k: np.array(x) for k, x in obs.items()})
    return np.load(fs, mmap_mode='r'), dict(np.load(fo))


def make_arm(arm, N):
    if arm == 'svd':
        return 'svd', None
    import spcfast
    p = arm.split(':')[1:]
    taus = [] if p[3] == 'none' else [float(x) for x in p[3].split('-')]
    return 'spcf', spcfast.SPCFast('ising', N, a=int(p[0]), fw=float(p[1]), iters=int(p[2]), taus=taus, ks=tuple(int(x) for x in p[4].split('-')),
                                   f_min=float(p[5]), every=int(p[6]), eps_min=float(p[7]), pattern=p[8], rel_skip=float(p[9]))


def run_floquet(label, N, depth, arm, chi, ref_states=None, verbose=True):
    thJ, thh = SETTINGS[label]
    gates = {d: make_kicked_gates(N, thJ, thh, d) for d in 'RL'}
    kind, cut = make_arm(arm, N)
    fc = L.FidCut(kind, cut)
    Tm = all_up_mps(N)
    if cut is not None and hasattr(cut, 'start'):
        cut.start(Tm)
    out = {k: [] for k in ('t', 'single', 'nn', 'nnn', 'z', 'zz', 'ov', 'logF', 'fired', 'calls')}
    c0 = time.process_time()
    t_tebd = 0.0
    for step in range(1, depth + 1):
        s0 = time.process_time()
        d = direction_of(step)
        for b in sweep_order(N, d):
            th = np.tensordot(Tm[b], Tm[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', gates[d][b], th)
            A = Tm[b - 1] if b >= 1 else None
            B = Tm[b + 2] if b + 2 <= N - 1 else None
            Tm[b], Tm[b + 1] = fc(th, chi, d, A, B, b)
        t_tebd += time.process_time() - s0
        v = M.mps_to_dense(Tm)
        v = v / np.linalg.norm(v)
        o = L.observables(v, N, 'ising')
        out['t'].append(step)
        for k in ('single', 'nn', 'nnn', 'z', 'zz'):
            out[k].append(o[k])
        out['ov'].append(np.vdot(np.asarray(ref_states[step - 1]) / np.linalg.norm(ref_states[step - 1]), v) if ref_states is not None else 0.0)
        out['logF'].append(fc.logF)
        out['fired'].append(getattr(cut, 'fired', 0))
        out['calls'].append(getattr(cut, 'calls', fc.ncut))
    meta = dict(label=label, thJ=thJ, thh=thh, N=N, depth=depth, arm=arm, chi=chi, cpu_tebd=t_tebd, cpu_total=time.process_time() - c0,
                fired=getattr(cut, 'fired', None), calls=getattr(cut, 'calls', None), ncut=fc.ncut)
    return Tm, {k: np.array(v) for k, v in out.items()}, meta


def selftest():
    # 1. a sweep equals U = prod ZZ prod RX built densely from Kronecker products (N = 6, both directions)
    N, thJ, thh = 6, -np.pi / 4, 0.7
    rng = np.random.default_rng(1)
    v0 = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
    v0 /= np.linalg.norm(v0)

    def op(P, i):
        return np.kron(np.kron(np.eye(2 ** i), P), np.eye(2 ** (N - i - 1)))
    Ukick = np.eye(2 ** N, dtype=complex)
    for i in range(N):
        Ukick = op(rx(thh), i) @ Ukick
    Uzz = np.eye(2 ** N, dtype=complex)
    for b in range(N - 1):
        Uzz = sla.expm(-1j * thJ / 2 * op(Z, b) @ op(Z, b + 1)) @ Uzz
    Uex = Uzz @ Ukick
    for d in 'RL':
        gates = make_kicked_gates(N, thJ, thh, d)
        v = v0.copy()
        for b in sweep_order(N, d):
            v = M.dense_gate(v, gates[b], b, N)
        err = np.abs(v - Uex @ v0).max()
        print(f'sweep {d} vs exact Floquet unitary: max err {err:.2e}')
        assert err < 1e-12
    # 2. the MPS sweep reproduces the dense sweep when chi is large enough (all-up, N = 8, depth 3, chi = 16 = exact)
    N = 8
    lab = 'A'
    ref_s, ref_o = ref_trajectory(lab, N, 3)
    for arm in ('svd', 'spcf:2:0:4:none:1-2-3:0:1:1e-7:all:1e-2'):
        Tm, arrs, meta = run_floquet(lab, N, 3, arm, 16, ref_s, verbose=False)
        e = np.abs(arrs['nn'][-1] - ref_o['nn'][-1]).max()
        print(f'{arm}: exact-chi max nn error {e:.2e}, |ov|^2 = {abs(arrs["ov"][-1]) ** 2:.12f}')
        assert e < 1e-10
    # 3. truncated runs differ from the reference and the spcf rule with taus=() fires
    N = 10
    ref_s, ref_o = ref_trajectory(lab, N, 6)
    for arm in ('svd', 'spcf:2:0:4:none:1-2-3:0:1:1e-7:all:1e-2'):
        Tm, arrs, meta = run_floquet(lab, N, 6, arm, 4, ref_s, verbose=False)
        e1 = np.sqrt(np.mean((arrs['single'][-1] - ref_o['single'][-1]) ** 2))
        print(f'{arm}: chi=4 depth 6: 1-site rms {e1:.3e}  infid {1 - abs(arrs["ov"][-1]) ** 2:.3e}  F_MPS {np.exp(arrs["logF"][-1]):.4f}  fired {meta["fired"]}/{meta["calls"]}')
    print('selftest OK')


if __name__ == '__main__':
    if sys.argv[1] == 'selftest':
        selftest()
        sys.exit(0)
    label, arm, chi, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    depth = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    rs, ro = ref_trajectory(label, N, depth)
    t0 = time.time()
    Tm, arrs, meta = run_floquet(label, N, depth, arm, chi, rs)
    meta['wall_total'] = time.time() - t0
    np.savez(out, meta=json.dumps(meta), **arrs)
    for i in range(depth):
        e1 = float(np.sqrt(np.mean((arrs['single'][i] - ro['single'][i]) ** 2)))
        ez = abs(float(arrs['z'][i][N // 2] - ro['z'][i][N // 2]))
        print(f"  d={i + 1:2d} 1site={e1:.2e} |dZ_mid|={ez:.2e} infid={1 - abs(arrs['ov'][i]) ** 2:.2e} F={np.exp(arrs['logF'][i]):.4f} fired={arrs['fired'][i]}/{arrs['calls'][i]}", flush=True)
    print('done', json.dumps(meta), flush=True)
