"""Shared helpers for the rank7 scripts: the SVD ladder of the cell, and the log-log interpolation that turns an error into a chi."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MATCHED_HP = ROOT / 'results' / 'high_pressure' / 'matched_hp.json'
METRICS = ('nn_rms', 'single_rms', 'nnn_rms', 'infid')


def svd_ladder(path=MATCHED_HP):
    """chi -> row of the SVD arm of the N = 20, T = 8 cell (chi 24 ... 192), ascending in chi."""
    rows = [r for r in json.load(open(path)) if r['arm'] == 'svd']
    rows.sort(key=lambda r: r['chi'])
    return rows


def chi_eq(chis, errs, e):
    """Chi at which the SVD error curve (log-log piecewise linear, errors decreasing in chi) reaches the error e.

    Returns (chi, flag) with flag '' inside the ladder, 'extrap_lo' when e is above the smallest-chi error (extrapolated with
    the first segment) and 'beyond' when e is below the largest-chi error (a lower bound: the last chi is returned)."""
    chis, errs = np.asarray(chis, float), np.asarray(errs, float)
    errs = np.minimum.accumulate(errs)
    if e < errs[-1]:
        return float(chis[-1]), 'beyond'
    lx, ly = np.log(chis), np.log(errs)
    flag = ''
    if e > errs[0]:
        flag = 'extrap_lo'
        j = 0
    else:
        j = int(np.searchsorted(-errs, -e) - 1)
        j = min(max(j, 0), len(chis) - 2)
    slope = (lx[j + 1] - lx[j]) / (ly[j + 1] - ly[j]) if ly[j + 1] != ly[j] else 0.0
    return float(np.exp(lx[j] + slope * (np.log(e) - ly[j]))), flag
