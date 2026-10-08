"""Apply the pre-registered T1 criteria (rank 8 on the non-local proxies, rank 9 on the local controls) to results/t1_*.json.

    python rank8_9/analyze_t1.py            # writes rank8_9/results/t1_summary.txt
"""
import glob
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
LO, HI = 1e-4, 1e-2
SPCF_ARMS = ('spcf', 'spcf_fw', 'spcf_ext', 'spcf_ext_fw')
LOCAL = ('tfim_ising', 'tfim_crit', 'heis', 'tfimh0.7', 'tfimh1.5', 'xxz0.5')
PLAN_NONLOCAL = lambda n: n.startswith('ppp_mo') or n.startswith('lr0.5') or n.startswith('lr1.5')


def t0_pass():
    """cells with f_loc^E(span<=4) >= 0.4 at some usable chi (infidelity in [1e-4,1e-2]) in the T0 sweep"""
    ok = {}
    for f in sorted(glob.glob(str(RES / 't0_[A-Z].json'))):
        for name, rec in json.load(open(f)).items():
            ok[name] = any(LO <= r['infid'] <= HI and r['locE4'] >= 0.4 for r in rec['rows'])
    return ok


def load():
    cells = {}
    for f in sorted(glob.glob(str(RES / 't1_*.json'))):
        for name, rec in json.load(open(f)).items():
            c = cells.setdefault(name, dict(info=rec['info'], E_ex=rec['E_ex'], runs={}))
            for r in rec['runs']:
                c['runs'][(r['chi'], r['arm'])] = r
    return cells


def med(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else float('nan')


def main():
    cells = load()
    out = []
    P = out.append
    P('T1 summary.  ratios > 1 mean the spcf variant is better than the best of SVD and variational fit (rank 8) / SVD and DMRG (rank 9).')
    # ------------------------------------------------------------------ rank 8: non-local proxies
    P('')
    P('RANK 8: non-local proxies.  rE = best(dE_svd, dE_var) / dE_arm ; rH = best(hw2_svd, hw2_var) / hw2_arm ; far = rms_far4_arm / rms_far4_svd')
    P(f"{'cell':<13}{'chi':>4}{'infid_svd':>10}{'arm':<13}{'rE':>7}{'rH':>7}{'far':>7}{'leak':>9}{'leak_svd':>10}{'fired':>6}{'wall/svd':>9}")
    passed = t0_pass()
    entries = {a: [] for a in SPCF_ARMS}
    extra = {a: [] for a in SPCF_ARMS}
    for name, c in cells.items():
        if name in LOCAL:
            continue
        chis = sorted({k[0] for k in c['runs']})
        for chi in chis:
            svd = c['runs'].get((chi, 'svd'))
            var = c['runs'].get((chi, 'varfit'))
            if svd is None or var is None:
                continue
            in_win = LO <= svd['infid'] <= HI
            bestE = min(abs(svd['dE']), abs(var['dE']))
            bestH = min(svd['hw2'], var['hw2'])
            P(f"{name:<13}{chi:>4}{svd['infid']:>10.2e} {'svd':<12}{'':>21}{'':>7}{svd.get('leak', float('nan')):>9.1e}{'':>10}{'':>6}{1.0:>9.1f}   {'(in window)' if in_win else '(outside window)'}")
            P(f"{'':<13}{'':>4}{var['infid']:>10.2e} {'varfit':<12}{abs(svd['dE']) / abs(var['dE']):>7.2f}{svd['hw2'] / var['hw2']:>7.2f}{var['rms_far4'] / svd['rms_far4']:>7.2f}{var.get('leak', float('nan')):>9.1e}")
            for a in SPCF_ARMS:
                r = c['runs'].get((chi, a))
                if r is None:
                    continue
                rE, rH, far = bestE / abs(r['dE']), bestH / r['hw2'], r['rms_far4'] / svd['rms_far4']
                leak, leak_svd = r.get('leak', 0.0), svd.get('leak', 0.0)
                P(f"{'':<13}{'':>4}{r['infid']:>10.2e} {a:<12}{rE:>7.2f}{rH:>7.2f}{far:>7.2f}{leak:>9.1e}{leak_svd:>10.1e}{r.get('fired', 0):>6}{r['wall'] / max(svd['wall'], 1e-9):>9.0f}")
                if in_win and (leak_svd <= 1e-10 or not np.isfinite(leak_svd)):
                    (entries if (PLAN_NONLOCAL(name) and passed.get(name, False)) else extra)[a].append((name, chi, rE, rH, far, leak))
    P('')
    for title, ent in (('HEADLINE: plan non-local proxies (MO-basis PPP, long-range Ising alpha<=1.5) whose T0 f_loc^E >= 0.4 at a usable chi', entries),
                       ('EXTRA: all other non-local cells (site-basis PPP, alpha=3, cells that failed the T0 0.4 test), same in-window/SVD-leak-free filter', extra)):
        P('Rank 8 criteria, ' + title)
        for a in SPCF_ARMS:
            e = ent[a]
            if not e:
                P(f'  {a:<12}: no entries')
                continue
            rE, rH, far, leak = [x[2] for x in e], [x[3] for x in e], [x[4] for x in e], [x[5] for x in e]
            succ = med(rE) >= 1.5 and med(rH) >= 1.5 and med(far) <= 1.10 and max(leak) <= 1e-10
            kill = med(rE) < 1.2 and med(rH) < 1.2
            anyok = [x[:2] for x in e if x[2] >= 1.5 and x[3] >= 1.5]
            P(f'  {a:<12}: n={len(e)}  median rE={med(rE):.3f} rH={med(rH):.3f} far={med(far):.3f}  max leak={max(leak):.1e}  '
              f'max rE={max(rE):.2f} max rH={max(rH):.2f}  entries with rE,rH>=1.5: {anyok if anyok else "none"}  -> {"SUCCESS" if succ else ("KILL" if kill else "AMBIGUOUS")}')
        P('')
    # ------------------------------------------------------------------ rank 9: local controls
    P('')
    P('RANK 9: local controls.  ratio = best(err_svd, err_dmrg) / err_spcf per observable class (rms over sites);  dE ratio = dE_svd / dE_spcf')
    P(f"{'cell':<12}{'chi':>4}{'infid_svd':>10}{'arm':<13}{'<X>':>8}{'<ZZ1>':>8}{'<ZZ3>':>8}{'dE':>8}")
    ent9 = {a: {'obsX': [], 'obsZZ1': [], 'obsZZ3': [], 'dE': []} for a in SPCF_ARMS}
    for name in LOCAL:
        if name not in cells:
            continue
        c = cells[name]
        for chi in sorted({k[0] for k in c['runs']}):
            svd, dm = c['runs'].get((chi, 'svd')), c['runs'].get((chi, 'dmrg'))
            if svd is None or dm is None:
                continue
            in_win = LO <= svd['infid'] <= HI
            P(f"{name:<12}{chi:>4}{svd['infid']:>10.2e} {'dmrg/svd':<12}" + ''.join(f"{svd[k] / dm[k]:>8.2f}" for k in ('obsX', 'obsZZ1', 'obsZZ3')) + f"{abs(svd['dE']) / abs(dm['dE']):>8.2f}   {'(in window)' if in_win else '(outside window)'}")
            for a in SPCF_ARMS:
                r = c['runs'].get((chi, a))
                if r is None:
                    continue
                rr = {k: min(svd[k], dm[k]) / r[k] for k in ('obsX', 'obsZZ1', 'obsZZ3')}
                rr['dE'] = abs(svd['dE']) / abs(r['dE'])
                P(f"{'':<12}{'':>4}{'':>10} {a:<12}" + ''.join(f"{rr[k]:>8.2f}" for k in ('obsX', 'obsZZ1', 'obsZZ3', 'dE')))
                if in_win:
                    for k in rr:
                        ent9[a][k].append(rr[k])
    P('')
    P('Rank 9 criteria over in-window entries of the local controls')
    for a in SPCF_ARMS:
        d = ent9[a]
        if not d['obsX']:
            P(f'  {a:<12}: no entries')
            continue
        allr = d['obsX'] + d['obsZZ1'] + d['obsZZ3']
        succ = med(allr) >= 1.5 and med(d['dE']) >= 1.0
        kill = all(med(d[k]) < 1.0 for k in ('obsX', 'obsZZ1', 'obsZZ3'))
        P(f'  {a:<12}: n={len(d["obsX"])}  median ratio all={med(allr):.3f}  X={med(d["obsX"]):.3f} ZZ1={med(d["obsZZ1"]):.3f} ZZ3={med(d["obsZZ3"]):.3f}  '
          f'median dE ratio={med(d["dE"]):.3f}  -> {"SUCCESS" if succ else ("KILL" if kill else "AMBIGUOUS")}')
    txt = '\n'.join(out)
    (RES / 't1_summary.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
