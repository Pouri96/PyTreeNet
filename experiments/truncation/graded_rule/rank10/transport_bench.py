"""Rank 10 transport bench.

  static  : Test 1.  One cut of the exact Heisenberg operator O(t) = U^dag eps_c U, all arms, class-resolved coefficient errors.
              python rank10/transport_bench.py static STEP CUT OUT.json [ARMS comma list]   (STEP * 0.1 = t, CUT = bond b between sites b, b+1)
  dynamic : Test 2.  Strang MPO evolution of eps_c (or Z_c) with the cut applied at every bond, scored against the dense reference.
              python rank10/transport_bench.py dyn KIND(E|Z) OUT.json CHIS ARMS
Arm specs: svd | rw:G | dmt:1 | dmt:2 | wsvd:LAM1:LAM2.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import json
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pmpo as P  # noqa: E402
import wsvd_op as W  # noqa: E402
import static_cut as SC  # noqa: E402
import heis_ref as H  # noqa: E402

MODEL, N, CSITE, DT = 'isingT', 12, 5, 0.1
CHIS_STATIC = [8, 16, 24, 32, 48]
LAMS = [1.0, 3.0, 10.0, 100.0]


def default_static_arms():
    arms = ['svd', 'rw:1.3', 'rw:1.6', 'rw:2.0', 'dmt:1', 'dmt:2']
    arms += [f'wsvd:{l:g}:0' for l in LAMS]                                   # n = 1 only
    arms += [f'wsvd:0:{l:g}' for l in LAMS]                                   # n = 2 only (nested: contains the near-1 block)
    arms += [f'wsvd:{a:g}:{b:g}' for a in LAMS for b in LAMS]                 # graded
    arms += ['wsvd:10000:0', 'wsvd:0:10000', 'wsvd:10000:10000']              # near-hard-constraint diagnostics (lam = 1e8 is ill-conditioned)
    return arms


def static_main(step, cut, out, arms=None):
    c = np.load(H.CACHE / f'snap_{MODEL}_N{N}_c{CSITE}_E_n20_dt{DT}_s{step}.npy')
    t0 = time.time()
    sc = SC.StaticCut(c, N, cut, MODEL, verbose=True)
    c1, c2 = P.local_coeffs_dense(c, N)
    rec = dict(model=MODEL, N=N, c=CSITE, step=step, t=round(step * DT, 6), cut=cut, k=sc.k, t_svd=sc.t_svd,
               fro2=sc.fro2, M2=sc.M2.tolist(), C_exact=P.energy_density(MODEL, c1, c2).tolist(), reserved=[sc.rL, sc.rR], arms={})
    for spec in (arms or default_static_arms()):
        ta = time.time()
        kind, rest = spec.split(':')[0], spec.split(':')[1:]
        chis = [x for x in CHIS_STATIC if not (kind == 'dmt' and x < 2 * 4 ** int(rest[0]))]
        if not chis:
            continue
        try:
            Pm, Qm, info = sc.arm_factors(spec, max(chis))
            tabs = sc.tables(Pm, Qm, chis)
        except W.InfeasibleCut as e:
            print(f'    {spec}: infeasible ({e})', flush=True)
            continue
        rec['arms'][spec] = dict(info=info, res={str(x): tabs[x] for x in chis}, wall=time.time() - ta)
        sp = {x: SC.span_classes(tabs[x]['S']) for x in chis}
        print(f'    {spec:<22s} {time.time() - ta:5.1f}s it={info["iters"]:>3d}{"" if info["converged"] else " NOCONV"}  ' + ' '.join(
            f'[{x}: fro={np.sqrt(tabs[x]["fro2"]):.1e} le3={np.sqrt(sp[x]["le3"]):.1e} s45={np.sqrt(sp[x]["s4"] + sp[x]["s5"]):.1e}]' for x in chis), flush=True)
        json.dump(rec, open(out, 'w'))
    rec['wall'] = time.time() - t0
    json.dump(rec, open(out, 'w'))
    print(f'  done t={rec["t"]} cut={cut} k={sc.k} wall={rec["wall"]:.0f}s', flush=True)


# ================================================================================================ dynamic mode (Test 2 / Test 3 building block)
SNAP_STEPS = (5, 10, 15, 20)


def mpo_norm2(T):
    x = np.ones((1, 1))
    for t in T:
        x = np.einsum('ab,apc,bpd->cd', x, t, t)
    return float(x[0, 0])


def dyn_one(kind, spec, chi, nsteps=20, ref_snaps=None, cut_for_classes=5):
    """One Strang MPO run.  Returns dict with local-coefficient histories c1 (n+1,N,4), c2 (n+1,N-1,4,4), norm2, and snapshot class tables."""
    gamma = W.spec_gamma(spec)
    cut = W.make_cut(spec)
    T0 = H.start_mpo(MODEL, N, CSITE, kind)
    c1 = np.zeros((nsteps + 1, N, 4))
    c2 = np.zeros((nsteps + 1, N - 1, 4, 4))
    n2 = np.zeros(nsteps + 1)
    c1[0], c2[0] = P.local_coeffs(T0)
    n2[0] = mpo_norm2(T0)
    snaps = {}
    lay = SC.static_layout(N, cut_for_classes)
    A, C = lay['nL'] + 1, lay['nR'] + 1
    Ar = (lay['a'][None, :] == np.arange(A)[:, None]).astype(float)
    Ac = (lay['c'][None, :] == np.arange(C)[:, None]).astype(float)

    def snap(n, T):
        Tp = P.unweight(T, gamma)
        c1[n], c2[n] = P.local_coeffs(Tp)
        n2[n] = mpo_norm2(Tp)
        if ref_snaps is not None and n in ref_snaps:
            ca = P.dense_op(Tp)
            cr = np.load(ref_snaps[n])
            E = (ca - cr).reshape(lay['nrow'], lay['ncol'])
            S = Ar @ ((E * E) @ Ac.T)
            snaps[n] = dict(S=S.tolist(), fro2=float(np.sum(E * E)), dot=float(ca @ cr), n2a=float(ca @ ca), n2r=float(cr @ cr))
            del ca, cr, E

    try:
        r = P.run_heis(MODEL, N, T0, chi, nsteps, DT, cut, gamma=gamma, snap=snap)
    except W.InfeasibleCut as e:
        return dict(spec=spec, kind=kind, chi=chi, infeasible=str(e))
    return dict(spec=spec, kind=kind, chi=chi, c1=c1.tolist(), c2=c2.tolist(), norm2=n2.tolist(), snaps={str(k): v for k, v in snaps.items()},
                wall=r['wall'], maxbond=r['maxbond'].tolist(), disc=r['disc'].tolist(), reserved=getattr(cut, 'reserved', None) and [list(x) for x in cut.reserved[-40:]])


def dyn_main(kind, out, chis, arms):
    c1, c2, n2, sn = H.cached_reference(MODEL, N, CSITE, kind, 20, DT, list(SNAP_STEPS))
    res = dict(model=MODEL, N=N, c=CSITE, kind=kind, dt=DT, ref=dict(c1=c1.tolist(), c2=c2.tolist(), norm2=n2.tolist()), runs=[])
    for spec in arms:
        for chi in chis:
            t0 = time.time()
            rec = dyn_one(kind, spec, chi, 20, sn)
            res['runs'].append(rec)
            if 'infeasible' in rec:
                print(f'  {kind} {spec} chi={chi}: infeasible', flush=True)
            else:
                ce = P.energy_density(MODEL, np.array(rec['c1']), np.array(rec['c2'])) if False else None
                print(f'  {kind} {spec:<16s} chi={chi:>3d} {time.time() - t0:5.1f}s  fro_rel(t=2)={np.sqrt(rec["snaps"]["20"]["fro2"] / rec["snaps"]["20"]["n2r"]):.3e}', flush=True)
            json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'static':
        static_main(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split(',') if len(sys.argv) > 5 else None)
    elif mode == 'dyn':
        dyn_main(sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(',')], sys.argv[5].split(','))
    else:
        raise SystemExit('unknown mode')
