"""Pooled table of the lookahead truncation methods against SVD at the same final bond dimension.

    python final_table.py
Each entry names a result file and arm, and the baseline row (classic per-gate SVD 'svd', peak = chi, or delayed SVD 'dsvd:f',
peak = f*chi). Ratios are method / baseline, so lower is better. Stored parameters are equal by construction and printed.
"""
import json

R = 'results/'
# (label, file, arm prefix, chi, baseline file, baseline arm, baseline chi)
CELLS = [
    # global fit with region lookahead, 20 times 0.1..2.0, windows 1-3
    ('global fit  Ising N=12 T=4  ', 'native_ising_region_t20.json', 'mfcr:2', 8, 'native_ising_wrank.json', 'dsvd:2', 8),
    ('global fit  DW    N=12 T=4  ', 'native_dw_t20.json', 'mfcr:2', 8, 'native_dw_region.json', 'dsvd:2', 8),
    ('global fit  ising2 N=12 T=4 ', 'native_ising2_t20.json', 'mfcr:2', 12, 'native_ising2_t20.json', 'dsvd:2', 12),
    ('global fit  Heis  N=12 T=1.5', 'native_heis_t20.json', 'mfcr:2', 12, 'native_heis_N12.json', 'dsvd:2', 12),
    # cut-local at peak chi
    ('cut-local   Ising N=12 T=4  ', 'lcut_ising12_grid.json', 'lcut:2:0.1', 8, 'native_ising_wrank.json', 'svd', 8),
    ('cut-local   DW    N=12 T=4  ', 'lcut_dw.json', 'lcut', 8, 'lcut_dw.json', 'svd', 8),
    ('cut-local   ising2 N=12 T=4 ', 'lcut_ising2.json', 'lcut', 12, 'lcut_ising2.json', 'svd', 12),
    ('cut-local   Heis  N=12 T=1.5', 'lcut_heis.json', 'lcut', 12, 'lcut_heis.json', 'svd', 12),
    ('cut-local   Ising N=16 T=4  ', 'n16_T4_lcut.json', 'lcut', 8, 'n16T4svd8', 'svd', 8),
    ('cut-local   Ising N=16 T=4  ', 'n16_T4_lcut.json', 'lcut', 12, 'n16T4svd12', 'svd', 12),
    ('cut-local   Ising N=16 T=5  ', 'n16_T5_lcut.json', 'lcut', 12, 'n16_T5_region.json', 'svd', 12),
    ('cut-local   Ising N=16 T=5  ', 'n16_T5_lcut.json', 'lcut', 16, 'n16_T5_region.json', 'svd', 16),
    ('cut-local   DW    N=16 T=5  ', 'n16_dw_T5_lcut.json', 'lcut', 12, 'n16_dw_T5_lcut.json', 'svd', 12),
    ('cut-local   Ising N=20 T=8  ', 'lcut_ising20_T8.json', 'lcut', 16, 'n20T8svd16', 'svd', 16),
    ('cut-local   ising2 N=16 T=4 ', 'n16_ising2_T4_lcut.json', 'lcut', 12, 'n16_ising2_T4_lcut.json', 'svd', 12),
    ('cut-local   Heis  N=16 T=1.5', 'n16_heis_T15_lcut.json', 'lcut', 12, 'n16_heis_T15_lcut.json', 'svd', 12),
    ('cut-local   Ising N=16 T=3  ', 'n16_T3_lcut.json', 'lcut', 8, 'n16_T3_region6.json', 'svd', 8),
    ('cut-local   Ising N=16 T=3  ', 'n16_T3_lcut.json', 'lcut', 10, 'n16_T3_region6.json', 'svd', 10),
    # global fit, N=16 (L=6, 5 taus, 100 iterations, working rank 1.5 chi), baseline svd (peak chi) and dsvd:1.5 (same working rank)
    ('global fit  Ising N=16 T=3  ', 'n16_T3_region6.json', 'mfcr:1.5', 8, 'n16_T3_region6.json', 'svd', 8),
    ('global fit  Ising N=16 T=3  ', 'n16_T3_region6.json', 'mfcr:1.5', 10, 'n16_T3_region6.json', 'svd', 10),
    ('  (vs dsvd:1.5)             ', 'n16_T3_region6.json', 'mfcr:1.5', 8, 'n16_T3_region6.json', 'dsvd:1.5', 8),
    ('  (vs dsvd:1.5)             ', 'n16_T3_region6.json', 'mfcr:1.5', 10, 'n16_T3_region6.json', 'dsvd:1.5', 10),
]


MANUAL = {   # baselines read from the bench logs where the json was overwritten by a later run
    'n16T4svd8': dict(chi=8, arm='svd', params=1448, peak=8, infid=1.156e-1, rdm2=5.54e-2, nn_rms=2.87e-2, E_abs=6.65e-2),
    'n16T4svd12': dict(chi=12, arm='svd', params=2856, peak=12, infid=2.695e-2, rdm2=1.97e-2, nn_rms=1.11e-2, E_abs=1.91e-2),
    'n20T8svd16': dict(chi=16, arm='svd', params=0, peak=16, infid=5.764e-1, rdm2=4.54e-2, nn_rms=2.29e-2, E_abs=4.19e-1),
}


def row(path, arm, chi):
    if path in MANUAL:
        return MANUAL[path]
    try:
        rows = json.load(open(R + path))
    except Exception:
        return None
    for r in rows:
        if r['chi'] == chi and (r['arm'] == arm or r['arm'].startswith(arm)):
            return r
    return None


print(f"{'method / cell':<30}{'chi':>4}{'params':>8}{'peak':>6}{'2-site':>8}{'nn corr':>9}{'energy':>8}{'infid':>7}   baseline")
for label, f, arm, chi, bf, barm, bchi in CELLS:
    m, b = row(f, arm, chi), row(bf, barm, bchi)
    if m is None or b is None:
        print(f"{label:<30}{chi:>4}   (missing: {'method' if m is None else 'baseline'})")
        continue
    print(f"{label:<30}{chi:>4}{m['params']:>8}{m['peak']:>6}{m['rdm2'] / b['rdm2']:>8.2f}{m['nn_rms'] / b['nn_rms']:>9.2f}"
          f"{m['E_abs'] / b['E_abs']:>8.2f}{m['infid'] / b['infid']:>7.2f}   {barm} (peak {b['peak']})")
