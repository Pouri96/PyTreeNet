"""T1 oracle factorisation probe for the closed-form whitened truncation (reports/first_moves/crosscut_whitening_novelty.md).

    python whiten/whiten_probe.py run  model N T chi out.json [stride] [max_cuts] [selftest_n]
    python whiten/whiten_probe.py summary out_a.json out_b.json ... [-o summary.json]

``run`` drives plain TEBD with the production spcf cut (SPCFast, best configuration a=2 fw=0 iters=4 taus={0,1.0} ks=1-2-3,
float32 closures), and for every cut that reaches the CG stage (rule/spcfast.py debug hook) it rebuilds the objective in float64
(wlib.CutLin) and scores these arms on the SAME exact objective f = ||F (h(rho_k) - h(rho))||^2:

  A0   SVD                                           A1p  spcf as run (production return value)
  A1r  replica of its CG (dense float64 J, diag-precond., 4 its, first acceptable alpha in (1,.5,.25))
  A2   exact Gauss-Newton step, dense lstsq           A2d  damped exact GN step, delta_rel in DELTAS
  A3   oracle Kronecker GN step: best Van Loan-Pitsianis fit of J^T J in tangent coordinates ((Re/Im x discarded) kron kept)
  A4   Shampoo-factor GN step: tangent first-order block of  Tr(dM^H L dM R)  with L=sum G G^H, R=sum G^H G, scale fitted to J^T J
  A5   closed-form whitened SVD in M-space with Shampoo factors, (I + mu L)^{1/2}, mu in MUS      (A5o: oracle VLP factors)
  A6   Kronecker-preconditioned CG (1 and 2 iterations), preconditioner = A3 or A4 Kronecker matrix

Per arm: f at the best alpha in (1, .5, .25) (tangent arms) or at alpha=1 (A5: closed form, no residual evaluation), span-2
residual ratio to SVD, discarded-weight ratio.  Retention rho = (f_svd - f_arm) / (f_svd - f_GN*), f_GN* = A2 (best alpha).

Pre-registered criteria are applied by ``summary`` (see the docstring of ``verdict``).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
import wlib as W

BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)
ALPHAS = (1.0, 0.5, 0.25)
DELTAS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
MUS = (0.1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
MUS_PREREG = (1.0, 1e1, 1e2, 1e3, 1e4)
A6_DELTAS = (1e-4, 1e-3, 1e-2)
ACCEPT = 1e-3


def tag(x):
    return f'{x:g}'


# ------------------------------------------------------------------------------------------------------------------- per cut
def analyze_cut(sp, theta, chi, dirn, b, res, fired, dbg, selftest=False):
    cl = W.CutLin(sp, theta, chi, dirn, b, M)
    k, nb, nx, l, r = cl.k, cl.nb, cl.nx, cl.l, cl.r
    t0 = time.perf_counter()
    rec = dict(b=int(b), dirn=dirn, l=l, r=r, k=k, nb=nb, nx=nx, nres=cl.nres, tail=cl.tail, f_svd=cl.f_svd, fired=bool(fired))
    rec['chk_fsvd'] = abs(cl.f_svd - dbg['f_svd']) / dbg['f_svd']
    span0 = max(cl.span2(cl.r0), 1e-300)
    fs = cl.f_svd
    Gam = cl.gamma_rows()
    JM = cl.jac_M(Gam)
    T = cl.T_dense()
    JC = JM @ T
    JtJ = JC.T @ JC
    bvec = -JC.T @ cl.r0
    sv = np.linalg.svd(JC, compute_uv=False)
    rec['J_rank'] = int(np.sum(sv > 1e-8 * sv[0])); rec['J_cond'] = float(sv[0] / max(sv[-1], 1e-300)); rec['J_s1'] = float(sv[0])

    if selftest:
        rng = np.random.default_rng(b + 17)
        st = {}
        x = rng.standard_normal(nx) * 0.02 / np.sqrt(max(nx, 1))
        st['jvp_rel'] = float(np.linalg.norm(dbg['jvp'](x) - JC @ x) / max(np.linalg.norm(JC @ x), 1e-300))
        g = rng.standard_normal(cl.nres)
        st['vjp_rel'] = float(np.linalg.norm(dbg['vjp'](g) - JC.T @ g) / max(np.linalg.norm(JC.T @ g), 1e-300))
        st['vjp_r0_rel'] = float(np.linalg.norm(dbg['vjp'](cl.r0) - JC.T @ cl.r0) / max(np.linalg.norm(JC.T @ cl.r0), 1e-300))
        xe = rng.standard_normal(nx) * 0.05 / np.sqrt(max(nx, 1))
        _, Mk = cl.retract(xe, 1.0)
        re_c = dbg['exact'](xe)[2]
        st['exact_rel'] = float(np.linalg.norm(re_c - cl.resid(Mk)) / max(np.linalg.norm(re_c), 1e-300))
        # finite differences of the M-space Jacobian (central, in a random complex direction)
        dM = rng.standard_normal((2 * l, 2 * r)) + 1j * rng.standard_normal((2 * l, 2 * r))
        dM *= 1e-5 / np.linalg.norm(dM)
        fd = (cl.F @ (cl.hnorm(cl.Mk0 + dM)[0] - cl.hnorm(cl.Mk0 - dM)[0])) / 2
        an = JM @ np.concatenate([dM.real.ravel(), dM.imag.ravel()])
        st['fd_rel'] = float(np.linalg.norm(fd - an) / max(np.linalg.norm(an), 1e-300))
        st['jvpM_rel'] = float(np.linalg.norm(cl.jvp_M(dM) - an) / max(np.linalg.norm(an), 1e-300))
        G_ = rng.standard_normal((cl.D, cl.no * cl.npp)) + 1j * rng.standard_normal((cl.D, cl.no * cl.npp))
        lhs = np.vdot(G_, cl.Wof(dM)); rhs = np.vdot(cl.Wadj(G_), dM)
        st['adj_rel'] = float(abs(lhs - rhs) / abs(lhs))
        rec['selftest'] = st

    def arm_from_Mk(Mk):
        f, r_ = cl.fval(Mk)
        return f, cl.span2(r_) / span0, cl.disc_ratio(Mk)

    def arm_tangent(x):
        """best alpha and first acceptable alpha for the tangent step x"""
        outs = []
        for al in ALPHAS:
            _, Mk = cl.retract(x, al)
            f, r_ = cl.fval(Mk)
            outs.append((f, cl.span2(r_) / span0, cl.disc_ratio(Mk), al))
        ib = int(np.argmin([o[0] for o in outs]))
        d = dict(f=outs[ib][0], sp=outs[ib][1], dc=outs[ib][2], al=outs[ib][3], f1=outs[0][0], sp1=outs[0][1])
        first = next((o for o in outs if o[0] < fs * (1 - ACCEPT)), None)
        d['ff'] = first[0] if first else fs
        d['spf'] = first[1] if first else 1.0
        return d

    arms = {}
    arms['A0'] = dict(f=fs, sp=1.0, dc=1.0, al=0.0, f1=fs, sp1=1.0, ff=fs, spf=1.0)
    # A1p: production result
    Qp = res[0].reshape(2 * l, k) if dirn == 'R' else res[1].reshape(k, 2 * r).conj().T
    f, s_, d_ = arm_from_Mk(cl.galerkin(Qp))
    arms['A1p'] = dict(f=f, sp=s_, dc=d_, al=1.0, f1=f, sp1=s_, ff=f, spf=s_)
    if fired:
        rec['chk_fc_log'] = abs(f - sp.log[-1][1]) / max(sp.log[-1][1], 1e-300)
    # A1r: CG replica
    Minv = cl.minv_diag()
    x = np.zeros(nx); res_ = bvec.copy(); z = Minv * res_; p = z.copy(); rz = res_ @ z; bn = np.linalg.norm(bvec)
    for _ in range(sp.iters):
        Ap = JtJ @ p
        al = rz / (p @ Ap)
        x += al * p; res_ -= al * Ap
        if np.linalg.norm(res_) < sp.cg_tol * bn:
            break
        z = Minv * res_; rzn = res_ @ z; p = z + (rzn / rz) * p; rz = rzn
    arms['A1r'] = arm_tangent(x)
    # A2 exact GN
    x2 = np.linalg.lstsq(JC, -cl.r0, rcond=1e-10)[0]
    arms['A2'] = arm_tangent(x2)
    lam, V = np.linalg.eigh(JtJ)
    lmax = lam[-1]
    Vb = V.T @ bvec
    for dl in DELTAS:
        xd = V @ (Vb / (np.maximum(lam, 0) + dl * lmax))
        arms[f'A2d_{tag(dl)}'] = arm_tangent(xd)
    # A3 oracle Kronecker (real, (Re/Im x discarded) kron kept)
    A3, B3, fid3 = W.vlp_rank1_real(JtJ, 2 * nb, k)
    K3 = np.kron(A3, B3)
    rec['kron_fid_tangent'] = fid3
    rec['kron_relerr_tangent'] = float(np.linalg.norm(JtJ - K3) / np.linalg.norm(JtJ))
    for dl in DELTAS:
        x3 = W.kron_solve(A3, B3, bvec, dl)
        arms[f'A3_{tag(dl)}'] = arm_tangent(x3)
    # Shampoo and oracle M-space factors
    Ls, Rs = W.shampoo(Gam)
    G1 = W.gram_G1(Gam)
    Lo, Ro, fidM = W.vlp_rank1_gamma(Gam, G1)
    rec['kron_fid_M'] = fidM
    # fit quality of the Shampoo and VLP product to the Hermitian Gauss-Newton form in M space (relative Frobenius error, scale fitted)
    for nm, (Lx, Rx) in (('sh', (Ls, Rs)), ('vlp', (Lo, Ro))):
        Kx = np.kron(Lx, Rx.T)                       # dM^H K dM = Tr(dM^H L dM R), row-major vec(dM)
        Gq = G1
        c = np.vdot(Kx, Gq).real / np.vdot(Kx, Kx).real
        rec[f'kron_relerr_M_{nm}'] = float(np.linalg.norm(Gq - c * Kx) / np.linalg.norm(Gq))
    # A4: Shampoo first-order tangent block
    A_ = cl.P1.conj().T @ Ls @ cl.P1
    B_ = cl.Q1 @ Rs @ cl.Q1.conj().T
    K0 = W.realify(np.kron(A_, B_.conj()))
    c4 = float(np.sum(JtJ * K0) / np.sum(K0 * K0))
    K4 = c4 * K0
    rec['kron_relerr_tangent_sh'] = float(np.linalg.norm(JtJ - K4) / np.linalg.norm(JtJ))
    l4, V4 = np.linalg.eigh(K4)
    Vb4 = V4.T @ bvec
    for dl in DELTAS:
        x4 = V4 @ (Vb4 / (np.maximum(l4, 0) + dl * l4[-1]))
        arms[f'A4_{tag(dl)}'] = arm_tangent(x4)
    # A5 / A5o closed-form whitened SVD
    for nm, (Lx, Rx) in (('A5', (Ls, Rs)), ('A5o', (Lo, Ro))):
        for mu in (0.0,) + MUS:
            if mu == 0.0 and nm == 'A5o':
                continue
            Lw, Rw = W.damped_pair(Lx, Rx, mu)
            Q, _ = W.whitened_subspace(cl.Mm, k, Lw, Rw, dirn)
            Mk = cl.galerkin(Q)
            f, s_, d_ = arm_from_Mk(Mk)
            e = dict(f=f, sp=s_, dc=d_, al=1.0, f1=f, sp1=s_)
            xq = cl.C_of_Q(Q)
            if xq is not None and mu > 0:
                eb = arm_tangent(xq)
                e.update(fb=eb['f'], spb=eb['sp'], dcb=eb['dc'], alb=eb['al'])
            else:
                e.update(fb=f, spb=s_, dcb=d_, alb=1.0)
            arms[f'{nm}_{tag(mu)}'] = e
    # A5 variants: which rows / which linearisation point feed the Shampoo factors (target-free ingredients)
    for nm, (at, rows) in (('A5s0', ('svd', 'static')), ('A5m', ('full', 'all')), ('A5sm', ('full', 'static'))):
        Gv = cl.gamma_rows(at=at, rows=rows)
        Lv, Rv = W.shampoo(Gv)
        for mu in (10.0, 1e2, 1e4):
            Lw, Rw = W.damped_pair(Lv, Rv, mu)
            Q, _ = W.whitened_subspace(cl.Mm, k, Lw, Rw, dirn)
            f, s_, d_ = arm_from_Mk(cl.galerkin(Q))
            arms[f'{nm}_{tag(mu)}'] = dict(f=f, sp=s_, dc=d_, al=1.0, f1=f, sp1=s_, fb=f, spb=s_, dcb=d_, alb=1.0)
    # A6 Kronecker-preconditioned CG on the exact J^T J
    for pn, Kp in (('A3', K3), ('A4', K4)):
        lk, Vk = np.linalg.eigh(0.5 * (Kp + Kp.T))
        for dl in A6_DELTAS:
            inv = Vk @ np.diag(1.0 / (np.maximum(lk, 0) + dl * lk[-1])) @ Vk.T
            x = np.zeros(nx); res_ = bvec.copy(); z = inv @ res_; p = z.copy(); rz = res_ @ z
            for it in (1, 2):
                Ap = JtJ @ p
                al = rz / (p @ Ap)
                x = x + al * p; res_ = res_ - al * Ap
                arms[f'A6{pn[1]}_{tag(dl)}_{it}'] = arm_tangent(x)
                if it == 1:
                    z = inv @ res_; rzn = res_ @ z; p = z + (rzn / rz) * p; rz = rzn
    rec['arms'] = arms
    rec['t_probe'] = time.perf_counter() - t0
    return rec


class ProbeCut(spcfast.SPCFast):
    def __init__(self, *a, stride=1, max_cuts=10 ** 9, selftest_n=0, **kw):
        super().__init__(*a, **kw)
        self.stride, self.max_cuts, self.selftest_n = stride, max_cuts, selftest_n
        self.records, self.seen = [], 0
        self.debug = []
        self.ckpt = None

    def __call__(self, theta, chi, dirn, A, B, b):
        self.debug = []
        f0 = self.fired
        res = super().__call__(theta, chi, dirn, A, B, b)
        dbg, self.debug = self.debug, []
        if dbg:
            self.seen += 1
            if (self.seen - 1) % self.stride == 0 and len(self.records) < self.max_cuts:
                rec = analyze_cut(self, theta, chi, dirn, b, res, self.fired > f0, dbg[0], selftest=len(self.records) < self.selftest_n)
                rec['seen'] = self.seen
                self.records.append(rec)
                if self.ckpt and len(self.records) % 10 == 0:
                    json.dump(dict(meta=dict(partial=True), records=self.records), open(self.ckpt, 'w'))
        return res


def run(model, N, T, chi, out, stride=1, max_cuts=10 ** 9, selftest_n=5):
    dt = 0.1
    n = int(round(T / dt))
    G = M.make_gates(model, N, dt)
    cut = ProbeCut(model, N, stride=stride, max_cuts=max_cuts, selftest_n=selftest_n, **BEST)
    cut.ckpt = out + '.partial'
    t0 = time.time()
    M.run_tebd(model, N, chi, n, dt, cut, gates=G)
    meta = dict(model=model, N=N, T=T, chi=chi, stride=stride, calls=cut.calls, fired=cut.fired, reached_cg=cut.seen, probed=len(cut.records),
                wall=time.time() - t0)
    json.dump(dict(meta=meta, records=cut.records), open(out, 'w'))
    print(meta, flush=True)


# ------------------------------------------------------------------------------------------------------------------- summary
def rho_of(rec, key, fkey='f'):
    a = rec['arms']
    fs, fg = a['A0']['f'], a['A2']['f']
    return (fs - a[key][fkey]) / (fs - fg)


def valid(rec):
    return (rec['arms']['A0']['f'] - rec['arms']['A2']['f']) > ACCEPT * rec['arms']['A0']['f']


def med(v):
    return float(np.median(v)) if len(v) else float('nan')


def summarize(recs, label):
    V = [r for r in recs if valid(r)]
    out = dict(label=label, n=len(recs), n_valid=len(V))
    if not V:
        return out
    pool = lambda key, fk='f': float(sum(r['arms']['A0']['f'] - r['arms'][key][fk] for r in V) / sum(r['arms']['A0']['f'] - r['arms']['A2']['f'] for r in V))
    rows = {}
    keys = [k for k in V[0]['arms'] if k != 'A0']
    for key in keys:
        rr = [rho_of(r, key) for r in V]
        row = dict(med=med(rr), q25=float(np.percentile(rr, 25)), q75=float(np.percentile(rr, 75)), pooled=pool(key),
                   span_med=med([r['arms'][key]['sp'] for r in V]), span_frac06=float(np.mean([r['arms'][key]['sp'] <= 0.6 for r in V])),
                   disc_med=med([r['arms'][key]['dc'] for r in V]))
        if key.startswith('A5'):
            rb = [rho_of(r, key, 'fb') for r in V]
            row.update(med_bestalpha=med(rb), span_med_b=med([r['arms'][key]['spb'] for r in V]), span_frac06_b=float(np.mean([r['arms'][key]['spb'] <= 0.6 for r in V])))
        if key == 'A1r':
            rf = [(r['arms']['A0']['f'] - r['arms'][key]['ff']) / (r['arms']['A0']['f'] - r['arms']['A2']['f']) for r in V]
            row['med_first'] = med(rf)
        # retention relative to production spcf (A1p)
        den = [(r['arms']['A0']['f'] - r['arms']['A1p']['f']) for r in V]
        ok = [i for i, d in enumerate(den) if d > ACCEPT * V[i]['arms']['A0']['f']]
        if ok:
            row['med_vs_A1p'] = med([(V[i]['arms']['A0']['f'] - V[i]['arms'][key]['f']) / den[i] for i in ok])
            row['n_vs_A1p'] = len(ok)
        rows[key] = row
    out['arms'] = rows
    out['stats'] = dict(
        gn_gain_med=med([(r['arms']['A0']['f'] - r['arms']['A2']['f']) / r['arms']['A0']['f'] for r in V]),
        a1p_gain_med=med([(r['arms']['A0']['f'] - r['arms']['A1p']['f']) / r['arms']['A0']['f'] for r in V]),
        a1p_span_med=med([r['arms']['A1p']['sp'] for r in V]),
        kron_fid_tangent_med=med([r['kron_fid_tangent'] for r in V]), kron_fid_M_med=med([r['kron_fid_M'] for r in V]),
        kron_relerr_tangent_med=med([r['kron_relerr_tangent'] for r in V]), kron_relerr_tangent_sh_med=med([r['kron_relerr_tangent_sh'] for r in V]),
        kron_relerr_M_sh_med=med([r['kron_relerr_M_sh'] for r in V]), kron_relerr_M_vlp_med=med([r['kron_relerr_M_vlp'] for r in V]),
        J_rank_over_nx_med=med([r['J_rank'] / r['nx'] for r in V]), nres_over_nx_med=med([r['nres'] / r['nx'] for r in V]),
        nx_med=med([r['nx'] for r in V]))
    return out


def best_key(S, prefix, cands, field='med'):
    sc = {c: S['arms'][f'{prefix}_{tag(c)}'][field] for c in cands if f'{prefix}_{tag(c)}' in S['arms']}
    return max(sc, key=lambda c: sc[c]), sc


def verdict(S):
    """Pre-registered (written before any T1 number was seen):
    rho3 = median rho(A3) at the best fixed delta_rel in DELTAS, rho5 = median rho(A5) (alpha=1, no residual evaluation) at the best
    fixed mu in MUS_PREREG, span = fraction of valid cuts with span-2 residual <= 0.6 x SVD for A5 at that mu.
    PASS      rho3 >= 0.6 and rho5 >= 0.5 and span >= 0.7
    KILL      rho3 < 0.35
    AMBIGUOUS otherwise (0.35 <= rho3 < 0.6, or rho3 passes but rho5 / span fail)."""
    d3, sc3 = best_key(S, 'A3', DELTAS)
    m5, sc5 = best_key(S, 'A5', MUS_PREREG)
    rho3 = S['arms'][f'A3_{tag(d3)}']['med']
    rho5 = S['arms'][f'A5_{tag(m5)}']['med']
    span = S['arms'][f'A5_{tag(m5)}']['span_frac06']
    if rho3 < 0.35:
        v = 'KILL'
    elif rho3 >= 0.6 and rho5 >= 0.5 and span >= 0.7:
        v = 'PASS'
    else:
        v = 'AMBIGUOUS'
    return dict(verdict=v, rho3=rho3, delta3=d3, rho5=rho5, mu5=m5, span_frac=span, scan3=sc3, scan5=sc5)


def summary(files, outfile=None):
    allrecs, per = [], {}
    for fn in files:
        d = json.load(open(fn))
        m = d['meta']
        lab = f"{m['model']}_N{m['N']}_T{m['T']}_chi{m['chi']}"
        per[lab] = d['records']
        allrecs += d['records']
    res = {}
    for lab, recs in list(per.items()) + [('POOLED', allrecs)]:
        S = summarize(recs, lab)
        if 'arms' in S:
            S['verdict'] = verdict(S)
            bulk = [r for r in recs if r['nx'] >= 50]
            Sb = summarize(bulk, lab + ' bulk(nx>=50)')
            if 'arms' in Sb:
                S['bulk'] = dict(n=Sb['n'], n_valid=Sb['n_valid'], verdict=verdict(Sb))
        res[lab] = S
    # cross-validated choices: pick delta/mu on one cell, evaluate on the other
    labs = list(per)
    if len(labs) == 2:
        cv = {}
        for tr, te in ((labs[0], labs[1]), (labs[1], labs[0])):
            d3, _ = best_key(res[tr], 'A3', DELTAS)
            m5, _ = best_key(res[tr], 'A5', MUS_PREREG)
            cv[f'{tr}->{te}'] = dict(delta3=d3, rho3_test=res[te]['arms'][f'A3_{tag(d3)}']['med'], mu5=m5, rho5_test=res[te]['arms'][f'A5_{tag(m5)}']['med'])
        res['crossval'] = cv
    # per-cut oracle choice of mu (not deployable; upper bound)
    V = [r for r in allrecs if valid(r)]
    if V:
        res['oracle_mu_A5'] = med([max((rho_of(r, f'A5_{tag(m)}') for m in MUS_PREREG)) for r in V])
        res['oracle_mu_A5o'] = med([max((rho_of(r, f'A5o_{tag(m)}') for m in MUS_PREREG)) for r in V])
    if outfile:
        json.dump(res, open(outfile, 'w'), indent=1)
    return res


def print_summary(res):
    for lab, S in res.items():
        if not isinstance(S, dict) or 'arms' not in S:
            continue
        print(f"\n=== {lab}: {S['n']} cuts, {S['n_valid']} with GN gain > {ACCEPT:g} f_svd")
        print('  stats', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in S['stats'].items()})
        v = S['verdict']
        print(f"  VERDICT {v['verdict']}: rho3 {v['rho3']:.3f} (delta {v['delta3']:g}), rho5 {v['rho5']:.3f} (mu {v['mu5']:g}), span<=0.6 frac {v['span_frac']:.2f}")
        if 'bulk' in S:
            print(f"  bulk n={S['bulk']['n']} valid={S['bulk']['n_valid']} verdict {S['bulk']['verdict']['verdict']} rho3 {S['bulk']['verdict']['rho3']:.3f} rho5 {S['bulk']['verdict']['rho5']:.3f}")
        print('  arm                  med rho   [q25,q75]       pooled  vsA1p  span2(med) frac<=.6  disc(med)')
        for key, row in S['arms'].items():
            extra = f"  | best-alpha rho {row['med_bestalpha']:.3f} span {row['span_med_b']:.2f}" if 'med_bestalpha' in row else ''
            print(f"  {key:18s} {row['med']:8.3f} [{row['q25']:7.3f},{row['q75']:7.3f}] {row['pooled']:8.3f} {row.get('med_vs_A1p', float('nan')):6.2f} {row['span_med']:9.2f} {row['span_frac06']:8.2f} {row['disc_med']:9.2f}{extra}")
    for kx in ('crossval', 'oracle_mu_A5', 'oracle_mu_A5o'):
        if kx in res:
            print(kx, res[kx])


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        model, N, T, chi, out = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
        stride = int(sys.argv[7]) if len(sys.argv) > 7 else 1
        mx = int(sys.argv[8]) if len(sys.argv) > 8 else 10 ** 9
        stn = int(sys.argv[9]) if len(sys.argv) > 9 else 5
        run(model, N, T, chi, out, stride, mx, stn)
    elif sys.argv[1] == 'summary':
        args = sys.argv[2:]
        o = None
        if '-o' in args:
            i = args.index('-o'); o = args[i + 1]; args = args[:i] + args[i + 2:]
        print_summary(summary(args, o))
