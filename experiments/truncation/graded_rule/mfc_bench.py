"""Marginal-fit compression bench, matched final bond dimension.

    PYLIBS=<dir with jax> python mfc_bench.py ising 20 8.0 0.1 svd,dsvd:2,mfc:2:0.01 16,24,32 out.json 3

Arms
  svd                 classic per-gate SVD truncation at chi (the baseline of every other bench)
  dsvd:f              gate sweep at working rank f*chi, one SVD compression to chi per step
  mfc:f:fw[:k:iters]  gate sweep at working rank f*chi, then a fit of all window marginals up to k sites
                      (default 3) plus fw*(1-F), L-BFGS for `iters` iterations (default 200)
  mfcl:f:fw:k:iters:ms:eps:kla[:lam:coarse:gamma:dmax]  mfc plus the Heisenberg-evolved Pauli strings of size kla (e.g. 1-2) at the
                      Trotter steps ms (e.g. 3-6), MPOs truncated at relative tolerance eps, weight lam; coarse > 0 makes ms count brickwork
                      Strang steps of time `coarse` instead (strict light cone, much smaller MPOs)
  mfcr:f:fw:k:iters:L:a:taus:kr[:lam_r]  mfc plus local-region lookahead (rule/region.py): window marginals (sizes kr, e.g. 1-2)
                      after evolving an L-site region (margin a) for the times taus (e.g. 0.5-1.0-2.0) under its own Hamiltonian
  lcut:a:fw:iters:taus:ks[:skip:ftol]  cut-local lookahead truncation (rule/lookcut.py): per-gate cut, region margin a, discarded-weight weight fw,
                      peak bond dimension = chi like classic TEBD
  spc:a:mode:rcond:passes:taus:ks  SVD cut plus a first-order restoration of the same cut-region marginals (rule/spccut.py), mode hard
                      (truncated-SVD solve, singular values below rcond * max dropped) or soft (Tikhonov), `passes` re-linearisations
  spcf:a:fw:iters:taus:ks[:fmin:every:eps:pattern:rel]  numpy matrix-free version of the spc cut (rule/spcfast.py): one Gauss-Newton step of the lcut
                      objective (discarded weight weight fw), `iters` preconditioned CG iterations, skip cuts with residual below fmin
                      or discarded weight fraction below eps, fire every `every`-th step, pattern all|R|L|alt (sweep direction / alternating bonds), rel = skip a cut whose discarded weight is below rel times the largest seen so far
Final states always have bond dimension <= chi. The transient working rank f*chi is recorded as `peak`.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
import numpy as np
import multiprocessing as mp
import _paths  # noqa: F401
import mpsenh as M
import hp_bench
import mfc


def one(args):
    model, N, T, dt, arm, chi = args
    nsteps = int(round(T / dt))
    ex = hp_bench.reference(model, N, T, dt)
    t0, c0 = time.time(), time.process_time()
    info = {}
    if arm == 'svd':
        Tm, _ = M.run_tebd(model, N, chi, nsteps, dt, M.svd_cut, gates=M.make_gates(model, N, dt))
        info = dict(peak=chi)
    else:
        p = arm.split(':')
        f = float(p[1])
        if p[0] == 'dsvd':
            Tm, info = mfc.run_mfc(model, N, chi, nsteps, dt, 'dsvd', chi_w_factor=f)
        elif p[0] == 'lcut':
            import lookcut
            a, fwc, iters = int(p[1]), float(p[2]), int(p[3])
            taus = [] if p[4] == 'none' else [float(x) for x in p[4].split('-')]
            ks = tuple(int(x) for x in p[5].split('-'))
            skip = float(p[6]) if len(p) > 6 else 0.0
            ftol = float(p[7]) if len(p) > 7 else 1e-14
            opt = p[8] if len(p) > 8 else 'scipy'
            rel = len(p) > 9 and p[9] == 'rel'
            edge = int(p[10]) if len(p) > 10 else 0
            cut = lookcut.LookCut(model, N, a=a, taus=taus, ks=ks, fw=fwc, maxiter=iters, skip_tol=skip, ftol=ftol,
                                  gtol=(1e-10 if ftol < 1e-12 else 1e-6) if opt == 'scipy' else 1e-4, opt=opt, rel=rel, edge=edge)
            Tm, _ = M.run_tebd(model, N, chi, nsteps, dt, cut, gates=M.make_gates(model, N, dt))
            info = dict(peak=chi, fired=cut.fired, calls=cut.calls, skipped=cut.skipped)
        elif p[0] == 'spc':
            import spccut
            a, mode, rc, passes = int(p[1]), p[2], float(p[3]), int(p[4])
            taus = [float(x) for x in p[5].split('-')]
            ks = tuple(int(x) for x in p[6].split('-'))
            cut = spccut.SPCCut(model, N, a=a, taus=taus, ks=ks, mode=mode, rcond=rc, passes=passes)
            Tm, _ = M.run_tebd(model, N, chi, nsteps, dt, cut, gates=M.make_gates(model, N, dt))
            lg = np.array(cut.log) if cut.log else np.zeros((1, 4))
            info = dict(peak=chi, fired=cut.fired, calls=cut.calls, res_ratio=float(np.median(lg[:, 1] / lg[:, 0])),
                        disc_ratio=float(np.median(lg[:, 3] / np.maximum(lg[:, 2], 1e-300))))
        elif p[0] == 'spcf':
            import spcfast
            a, fwc, iters = int(p[1]), float(p[2]), int(p[3])
            taus = [float(x) for x in p[4].split('-')]
            ks = tuple(int(x) for x in p[5].split('-'))
            fmin = float(p[6]) if len(p) > 6 else 0.0
            every = int(p[7]) if len(p) > 7 else 1
            eps = float(p[8]) if len(p) > 8 else 1e-10
            pat = p[9] if len(p) > 9 else 'all'
            rel = float(p[10]) if len(p) > 10 else 0.0
            prec = p[11] if len(p) > 11 else 'f32'
            import ast
            extra = {}
            for it in p[12:]:
                kk, vv = it.split('=')
                try:
                    extra[kk] = ast.literal_eval(vv)
                except Exception:
                    extra[kk] = vv
            cut = spcfast.SPCFast(model, N, a=a, taus=taus, ks=ks, fw=fwc, iters=iters, f_min=fmin, every=every, eps_min=eps, pattern=pat,
                                  rel_skip=rel, precision=prec, **extra)
            Tm, _ = M.run_tebd(model, N, chi, nsteps, dt, cut, gates=M.make_gates(model, N, dt))
            lg = np.array(cut.log) if cut.log else np.zeros((1, 4))
            info = dict(peak=chi, fired=cut.fired, calls=cut.calls, res_ratio=float(np.median(lg[:, 1] / np.maximum(lg[:, 0], 1e-300))),
                        disc_ratio=float(np.median(lg[:, 3] / np.maximum(lg[:, 2], 1e-300))))
            print('   spcf time split', {kk: round(v, 2) for kk, v in cut.tm.items()}, 'skipped', cut.skipped, 'matvecs', cut.nmv, flush=True)
        elif p[0] == 'mfcr':
            import lookahead, region
            fw, kmax, iters = float(p[2]), int(p[3]), int(p[4])
            L, a = int(p[5]), int(p[6])
            taus = [float(x) for x in p[7].split('-')]
            kr = tuple(int(x) for x in p[8].split('-'))
            lam_r = float(p[9]) if len(p) > 9 else 1.0
            reg = region.RegionLookahead(model, N, L, a, taus, ks=kr)
            Tm, info = lookahead.run_mfcl(model, N, chi, nsteps, dt, chi_w_factor=f, ks=tuple(range(1, kmax + 1)), fw=fw,
                                          maxiter=iters, ms=(), region=reg, lam_r=lam_r)
        elif p[0] == 'mfcl':
            import lookahead
            fw, kmax, iters = float(p[2]), int(p[3]), int(p[4])
            ms = tuple(int(x) for x in p[5].split('-'))
            eps = float(p[6])
            kla = tuple(int(x) for x in p[7].split('-'))
            lam = float(p[8]) if len(p) > 8 else 1.0
            coarse = float(p[9]) if len(p) > 9 else 0.0
            gamma = float(p[10]) if len(p) > 10 else 0.0
            dmax = int(p[11]) if len(p) > 11 else 150
            Tm, info = lookahead.run_mfcl(model, N, chi, nsteps, dt, chi_w_factor=f, ks=tuple(range(1, kmax + 1)), fw=fw,
                                          maxiter=iters, ms=ms, kla=kla, eps=eps, lam=lam, coarse=coarse, gamma=gamma, dmax=dmax)
        else:
            fw = float(p[2])
            kmax = int(p[3]) if len(p) > 3 else 3
            iters = int(p[4]) if len(p) > 4 else 200
            Tm, info = mfc.run_mfc(model, N, chi, nsteps, dt, 'mfc', chi_w_factor=f, ks=tuple(range(1, kmax + 1)),
                                   fw=fw, maxiter=iters)
    wall, cpu = time.time() - t0, time.process_time() - c0
    ap = M.mps_to_dense(Tm)
    e = M.errors(ex, ap, N, model)
    e.update(hp_bench.marginal_errors(ex, ap, N))
    _a, _b = M.local_obs(ex, N, model), M.local_obs(ap, N, model)
    e.update({k + '_max': float(np.max(np.abs(_a[k] - _b[k]))) for k in ('single', 'nn', 'nnn')})
    e.update(arm=arm, chi=chi, params=M.stored_params(Tm), wall=round(wall, 1), cpu=round(cpu, 1), peak=info.get('peak'))
    for kk in ('res_ratio', 'disc_ratio', 'fired', 'calls'):
        if kk in info:
            e[kk] = info[kk]
    if info.get('hist'):
        h = np.array(info['hist'])
        e.update(fit_f0=float(np.median(h[:, 0])), fit_f1=float(np.median(h[:, 1])), fit_nit=float(np.median(h[:, 2])))
    return e


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    arms, chis = sys.argv[5].split(','), [int(c) for c in sys.argv[6].split(',')]
    out, nproc = sys.argv[7], int(sys.argv[8]) if len(sys.argv) > 8 else 3
    hp_bench.reference(model, N, T, dt)
    jobs = [(model, N, T, dt, a, c) for c in chis for a in arms]
    res, t0 = [], time.time()
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(one, jobs):
            res.append(r)
            json.dump(res, open(out, 'w'))
            print(f"{len(res)}/{len(jobs)} chi={r['chi']} {r['arm']:>16} infid={r['infid']:.3e} rdm2={r['rdm2']:.2e} "
                  f"nn={r['nn_rms']:.2e} E={r['E_abs']:.2e} peak={r['peak']} {r['wall']:.0f}s [{int(time.time() - t0)}s]"
                  + (f" res_after/before={r['res_ratio']:.2f} disc_after/before={r['disc_ratio']:.2f} fired={r['fired']}/{r['calls']}" if 'res_ratio' in r else ''), flush=True)
