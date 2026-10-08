"""Inject-and-propagate test: how long does the error of ONE truncated Strang step live?

    python kick.py model N t0 chi tprop out.json [spcf kwargs json]
The state is evolved to t0 with a bond dimension large enough to be exact.  One full Strang step is then done from that exact MPS
with (a) the plain SVD cut at `chi` and (b) the spcf cut at `chi`; the exact step is the reference.  All three dense states are then
propagated EXACTLY (no truncation) for tprop, and the deviation of the observables from the exact trajectory is recorded each step.
Unitary evolution keeps every infidelity constant, so any change of the local-observable deviation is the dynamics moving the error
between local and non-local operators.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast

model, N, t0, chi, tprop = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
out = sys.argv[6]
kw = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
dt = 0.1
G = M.make_gates(model, N, dt)
n0, n1 = int(round(t0 / dt)), int(round(tprop / dt))


def sweep(Tm, cut, ch):
    tails = []
    for rng, d in ((range(N - 1), 'R'), (range(N - 2, -1, -1), 'L')):
        for b in rng:
            th = np.einsum('abst,lstr->labr', G[b], np.tensordot(Tm[b], Tm[b + 1], axes=([2], [0])))
            A = Tm[b - 1] if b >= 1 else None
            B = Tm[b + 2] if b + 2 <= N - 1 else None
            if cut is M.svd_cut:
                Tm[b], Tm[b + 1], _, _ = M.svd_cut(th, ch, d)
            else:
                Tm[b], Tm[b + 1], _, _ = cut(th, ch, d, A, B, b)


def dstep(v):
    for b in range(N - 1):
        v = M.dense_gate(v, G[b], b, N)
    for b in range(N - 2, -1, -1):
        v = M.dense_gate(v, G[b], b, N)
    return v


def devs(v, ve):
    o, oe = M.local_obs(v, N, model), M.local_obs(ve, N, model)
    d = {k: float(np.sqrt(np.mean((o[k] - oe[k]) ** 2))) for k in ('single', 'nn', 'nnn')}
    d['E'] = float(o['E'] - oe['E'])
    d['infid'] = float(1 - abs(np.vdot(v / np.linalg.norm(v), ve / np.linalg.norm(ve))) ** 2)
    return d


Tex, _ = M.run_tebd(model, N, 4096, n0, dt, M.svd_cut, gates=G)
print('exact bond dims at t0:', max(M.ranks(Tex)), flush=True)
cut = spcfast.SPCFast(model, N, **{**BEST, **kw})
Ts, Tc, Te = list(Tex), list(Tex), list(Tex)
cut.start(Tc)
sweep(Ts, M.svd_cut, chi)
sweep(Tc, cut, chi)
sweep(Te, M.svd_cut, 4096)
tails = [l[2] for l in cut.log]
print(f'spcf fired {cut.fired} of {cut.calls} cuts, mean discarded weight at fired cuts {np.mean([l[6] for l in cut.log]) if cut.log else 0:.2e}', flush=True)
vs, vc, ve = (M.mps_to_dense(T) for T in (Ts, Tc, Te))
rows = []
print('tprop |  1-site dev: svd  spcf  ratio | nn dev: svd  spcf  ratio | E dev: svd  spcf | infid svd spcf')
for s in range(n1 + 1):
    ds, dc = devs(vs, ve), devs(vc, ve)
    rows.append(dict(tprop=round(s * dt, 6), svd=ds, spcf=dc))
    print(f'{s * dt:4.1f}  | {ds["single"]:.2e} {dc["single"]:.2e} {dc["single"] / max(ds["single"], 1e-300):6.2f} | {ds["nn"]:.2e} {dc["nn"]:.2e} {dc["nn"] / max(ds["nn"], 1e-300):6.2f} | '
          f'{ds["E"]:+.1e} {dc["E"]:+.1e} | {ds["infid"]:.1e} {dc["infid"]:.1e}', flush=True)
    vs, vc, ve = dstep(vs), dstep(vc), dstep(ve)
json.dump(rows, open(out, 'w'))
