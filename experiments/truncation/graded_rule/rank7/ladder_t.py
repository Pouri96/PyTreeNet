"""Rank 7, T2: time-resolved TEBD ladders with free post-processing data, one arm per process.

    python rank7/ladder_t.py ising 20 8.0 svd 32 rank7/results/t2_svd_32.npz
    python rank7/ladder_t.py ising 20 8.0 spcf:2:0:4:1.0:1-2-3:0:1:1e-7:all:0.3 24 rank7/results/t2_spcf_g_24.npz
    python rank7/ladder_t.py ising 20 8.0 selftest 10 /dev/null          # checks against M.run_tebd on a small chain

The step loop is the one of ``mpsenh.run_tebd`` (Strang sweeps, same gates as the dense reference).  Every ``every`` steps
(default 5, i.e. dt * 5 = 0.5) the MPS is contracted to the dense vector and the following are stored:

  single, nn, nnn   the 3N + 9(N-1) + 9(N-2) Pauli expectation values of ``mpsenh.local_obs`` (so that the rms errors are
                    identical to ``mpsenh.errors``), as raw values, to be post-processed (rescaling, extrapolation)
  z, zz             <Z_i> and <Z_i Z_j> for all pairs (the ingredients of the D-Wave style error eps_c)
  ov                <ref|psi> of the normalised states, so infidelity = 1 - |ov|^2
  logF              running sum of log f_cut, f_cut = |<theta|theta'>|^2 / (|theta|^2 |theta'|^2) per cut, theta' = the returned
                    (A, B) contracted.  For SVD this is 1 - discarded weight; for any other cut it is measured the same way.
                    F_MPS = exp(logF) is the Zhou-Stoudenmire-Waintal / Mandra / Anand fidelity estimate.

The dense reference trajectory (states at the sample steps, 16 x 16.8 MB for N = 20) is built once with ``mpsenh.dense_gate`` and
cached in rank7/_refcache/ (ignored by git); its observables go to rank7/results/t2_ref_obs_*.npz.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import json
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)
import _paths  # noqa: E402,F401  (this repository's pytreenet and rule/ onto sys.path)
import mpsenh as M  # noqa: E402

CACHE = HERE / '_refcache'
RES = HERE / 'results'

_ZMAT = {}


def zmat(N):
    """(N, 2^N) float64 matrix of the +-1 eigenvalues of Z_i on the computational basis (site 0 = most significant bit)."""
    if N not in _ZMAT:
        idx = np.arange(2 ** N)
        _ZMAT[N] = np.stack([1.0 - 2.0 * ((idx >> (N - 1 - i)) & 1) for i in range(N)]).astype(np.float64)
    return _ZMAT[N]


def observables(v, N, model):
    """Pauli expectation values of ``mpsenh.local_obs`` plus <Z_i>, <Z_i Z_j> from the (normalised) dense vector."""
    v = v / np.linalg.norm(v)
    lo = M.local_obs(v, N, model)
    p = (v.real ** 2 + v.imag ** 2)
    Z = zmat(N)
    z = Z @ p
    zz = (Z * p[None, :]) @ Z.T
    return dict(single=lo['single'], nn=lo['nn'], nnn=lo['nnn'], z=z, zz=zz)


def sample_steps(nsteps, every):
    return [s for s in range(1, nsteps + 1) if s % every == 0]


def ref_paths(model, N, T, dt, every):
    tag = f'{model}_N{N}_T{T}_dt{dt}_e{every}'
    return CACHE / f'traj_{tag}.npy', RES / f't2_ref_obs_{tag}.npz'


def ref_trajectory(model, N, T, dt, every):
    """States of the dense Trotter reference at every sample step (cached), and their observables (cached)."""
    nsteps = int(round(T / dt))
    fs, fo = ref_paths(model, N, T, dt, every)
    steps = sample_steps(nsteps, every)
    if not (fs.exists() and fo.exists()):
        CACHE.mkdir(exist_ok=True)
        RES.mkdir(exist_ok=True)
        Gs = M.make_gates(model, N, dt)
        v = M.mps_to_dense(M.initial_mps(model, N))
        states = np.zeros((len(steps), 2 ** N), dtype=complex)
        obs = {k: [] for k in ('single', 'nn', 'nnn', 'z', 'zz')}
        k = 0
        for step in range(1, nsteps + 1):
            for b in range(N - 1):
                v = M.dense_gate(v, Gs[b], b, N)
            for b in range(N - 2, -1, -1):
                v = M.dense_gate(v, Gs[b], b, N)
            if step in steps:
                states[k] = v
                o = observables(v, N, model)
                for key in obs:
                    obs[key].append(o[key])
                k += 1
        np.save(fs, states)
        np.savez(fo, t=np.array(steps) * dt, **{key: np.array(val) for key, val in obs.items()})
    return np.load(fs, mmap_mode='r'), dict(np.load(fo))


class FidCut:
    """Wrap a cut so that every call also measures the truncation fidelity of the returned pair against theta."""
    def __init__(self, kind, cut):
        self.kind, self.cut = kind, cut
        self.logF = 0.0
        self.ncut = 0
        self.fmin = 1.0

    def __call__(self, th, chi, dirn, A, B, b):
        if self.kind == 'svd':
            Ta, Tb, _, _ = M.svd_cut(th, chi, dirn)
        else:
            Ta, Tb, _, _ = self.cut(th, chi, dirn, A, B, b)
        thp = np.tensordot(Ta, Tb, axes=([2], [0]))
        f = abs(np.vdot(th, thp)) ** 2 / (np.vdot(th, th).real * np.vdot(thp, thp).real)
        self.logF += float(np.log(f))
        self.fmin = min(self.fmin, float(f))
        self.ncut += 1
        return Ta, Tb


def make_arm(arm, model, N):
    """arm = 'svd' or 'spcf:a:fw:iters:taus:ks:fmin:every:eps:pattern:rel' (the OPT string of pareto_spcf.py)."""
    if arm == 'svd':
        return 'svd', None
    if arm.startswith('spcf:'):
        import spcfast
        p = arm.split(':')[1:]
        return 'spcf', spcfast.SPCFast(model, N, a=int(p[0]), fw=float(p[1]), iters=int(p[2]), taus=[float(x) for x in p[3].split('-')],
                                       ks=tuple(int(x) for x in p[4].split('-')), f_min=float(p[5]), every=int(p[6]),
                                       eps_min=float(p[7]), pattern=p[8], rel_skip=float(p[9]))
    raise ValueError(arm)


def run_arm(model, N, T, dt, every, arm, chi, ref_states=None, ref_obs=None, verbose=True):
    nsteps = int(round(T / dt))
    Gs = M.make_gates(model, N, dt)
    kind, cut = make_arm(arm, model, N)
    fc = FidCut(kind, cut)
    Tm = M.initial_mps(model, N)
    if cut is not None and hasattr(cut, 'start'):
        cut.start(Tm)
    out = {k: [] for k in ('t', 'single', 'nn', 'nnn', 'z', 'zz', 'ov', 'logF', 'fired', 'calls')}
    t_tebd = 0.0
    c0 = time.process_time()
    k = 0
    for step in range(1, nsteps + 1):
        s0 = time.process_time()
        for b in range(N - 1):
            th = np.tensordot(Tm[b], Tm[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Gs[b], th)
            A = Tm[b - 1] if b >= 1 else None
            B = Tm[b + 2] if b + 2 <= N - 1 else None
            Tm[b], Tm[b + 1] = fc(th, chi, 'R', A, B, b)
        for b in range(N - 2, -1, -1):
            th = np.tensordot(Tm[b], Tm[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Gs[b], th)
            A = Tm[b - 1] if b >= 1 else None
            B = Tm[b + 2] if b + 2 <= N - 1 else None
            Tm[b], Tm[b + 1] = fc(th, chi, 'L', A, B, b)
        t_tebd += time.process_time() - s0
        if step % every == 0:
            v = M.mps_to_dense(Tm)
            v = v / np.linalg.norm(v)
            o = observables(v, N, model)
            out['t'].append(step * dt)
            for key in ('single', 'nn', 'nnn', 'z', 'zz'):
                out[key].append(o[key])
            out['ov'].append(np.vdot(np.asarray(ref_states[k]) / np.linalg.norm(ref_states[k]), v) if ref_states is not None else 0.0)
            out['logF'].append(fc.logF)
            out['fired'].append(getattr(cut, 'fired', 0))
            out['calls'].append(getattr(cut, 'calls', fc.ncut))
            k += 1
            if verbose and ref_obs is not None:
                e1 = float(np.sqrt(np.mean((o['single'] - ref_obs['single'][k - 1]) ** 2)))
                e2 = float(np.sqrt(np.mean((o['nn'] - ref_obs['nn'][k - 1]) ** 2)))
                print(f"  t={step * dt:4.1f} 1site={e1:.2e} nn={e2:.2e} infid={1 - abs(out['ov'][-1]) ** 2:.2e} F={np.exp(fc.logF):.4f} "
                      f"fired={out['fired'][-1]}/{out['calls'][-1]}", flush=True)
    meta = dict(model=model, N=N, T=T, dt=dt, every=every, arm=arm, chi=chi, cpu_tebd=t_tebd, cpu_total=time.process_time() - c0,
                fired=getattr(cut, 'fired', None), calls=getattr(cut, 'calls', None), fmin=fc.fmin, ncut=fc.ncut)
    arrs = {k: np.array(v) for k, v in out.items()}
    return Tm, arrs, meta


def selftest(model='ising', N=10, T=1.0, dt=0.1, chi=8):
    """The bookkeeping loop must reproduce ``mpsenh.run_tebd`` exactly (both arms), and the dense observables must match errors()."""
    for arm in ('svd', 'spcf:2:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2'):
        kind, cut = make_arm(arm, model, N)
        Tref, _ = M.run_tebd(model, N, chi, int(round(T / dt)), dt, M.svd_cut if kind == 'svd' else cut, gates=M.make_gates(model, N, dt))
        Tm, arrs, meta = run_arm(model, N, T, dt, 5, arm, chi, ref_states=None, ref_obs=None, verbose=False)
        d = max(np.abs(a - b).max() for a, b in zip(Tm, Tref))
        print(f'selftest {arm}: max |T - T_run_tebd| = {d:.2e}   (must be 0)   F_MPS={np.exp(arrs["logF"][-1]):.6f}')
        assert d == 0.0
        v = M.mps_to_dense(Tm)
        ex = M.dense_reference(model, N, int(round(T / dt)), dt, gates=M.make_gates(model, N, dt))
        e = M.errors(ex, v, N, model)
        o, oe = observables(v, N, model), observables(ex, N, model)
        mine = {k: float(np.sqrt(np.mean((o[k] - oe[k]) ** 2))) for k in ('single', 'nn', 'nnn')}
        print('   errors() ', {k: round(e[k + '_rms'], 8) for k in ('single', 'nn', 'nnn')}, ' mine ', {k: round(x, 8) for k, x in mine.items()},
              ' infid=%.3e' % e['infid'], ' 1-|ov|^2 vs ref: n/a')
        for k in ('single', 'nn', 'nnn'):
            assert abs(e[k + '_rms'] - mine[k]) < 1e-12
        # zz sanity against a direct dense evaluation
        vv = v / np.linalg.norm(v)
        P = np.abs(vv.reshape([2] * N)) ** 2
        i, j = 1, 4
        sl = P.sum(axis=tuple(a for a in range(N) if a not in (i, j)))
        zz_ij = sl[0, 0] + sl[1, 1] - sl[0, 1] - sl[1, 0]
        assert abs(zz_ij - o['zz'][i, j]) < 1e-12
    print('selftest OK')


if __name__ == '__main__':
    model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    arm = sys.argv[4]
    if arm == 'selftest':
        selftest(model, N=int(sys.argv[5]))
        sys.exit(0)
    chi, out = int(sys.argv[5]), sys.argv[6]
    dt, every = 0.1, 5
    rs, ro = ref_trajectory(model, N, T, dt, every)
    t0 = time.time()
    Tm, arrs, meta = run_arm(model, N, T, dt, every, arm, chi, rs, ro)
    meta['wall_total'] = time.time() - t0
    np.savez(out, meta=json.dumps(meta), **arrs)
    print('done', json.dumps(meta), flush=True)
