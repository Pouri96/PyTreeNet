"""The chain graded rule (mpsenh.EnhCut) as the final truncation cut of RAGE_MPS.

RAGE_MPS truncates with one leftward sweep of per-bond cuts, centre on the node being cut.  Each cut
is the SVD of the two-site tensor theta = T_{b-1} T_b with a left-isometric T_{b-2} and a
right-isometric T_{b+1}, exactly the setting of EnhCut.  The wrapper replaces the library's
split_svd_contract_sv_to_neighbour by a call to EnhCut on theta.  kappa = 0 gives the plain SVD cut
through the same code path.
"""
import numpy as np
import mpsenh as M
import pytreenet.core.truncation.sweeping_truncation as ST
from pytreenet.util.tensor_splitting import truncate_singular_values

STATE = dict(cut=None, N=None, fired=0, cuts=0)
_ORIG = ST.split_svd_contract_sv_to_neighbour


def _idx(node_id):
    return int(node_id.replace('qubit', ''))


def to_lpr(ttn, node_id, N):
    """Node tensor as (left bond, phys, right bond), singleton bonds at the chain ends."""
    node = ttn.nodes[node_id]
    T = ttn.tensors[node_id]
    i = _idx(node_id)
    axes = {}
    if i > 0:
        axes['l'] = node.neighbour_index(f'qubit{i - 1}')
    if i < N - 1:
        axes['r'] = node.neighbour_index(f'qubit{i + 1}')
    axes['p'] = node.open_legs[0]
    perm = [axes[k] for k in ('l', 'p', 'r') if k in axes]
    T = np.transpose(T, perm)
    if 'l' not in axes:
        T = T[None]
    if 'r' not in axes:
        T = T[..., None]
    return T


def from_lpr(ttn, node_id, T3, N):
    node = ttn.nodes[node_id]
    i = _idx(node_id)
    keys = []
    if i > 0:
        keys.append(('l', node.neighbour_index(f'qubit{i - 1}')))
    keys.append(('p', node.open_legs[0]))
    if i < N - 1:
        keys.append(('r', node.neighbour_index(f'qubit{i + 1}')))
    if i == 0:
        T3 = T3[0]
    if i == N - 1:
        T3 = T3[..., 0]
    # T3 axes now follow the order of 'keys'; node order is by leg position
    order = np.argsort([pos for _, pos in keys])
    return np.transpose(T3, order)


def _cut(ttn, node_id, next_id, params, preserve_legs_order=False):
    N = STATE['N']
    i, j = _idx(node_id), _idx(next_id)
    if j != i - 1:
        return _ORIG(ttn, node_id, next_id, params, preserve_legs_order=preserve_legs_order)
    b = j                                                  # pair (b, b+1) = (next, node)
    Tn, Tc = to_lpr(ttn, next_id, N), to_lpr(ttn, node_id, N)
    theta = np.einsum('lpm,mqr->lpqr', Tn, Tc)
    nrm = float(np.linalg.norm(theta))
    theta = theta / nrm
    l, r = theta.shape[0], theta.shape[3]
    s = np.linalg.svd(theta.reshape(l * 2, 2 * r), compute_uv=False)
    k = len(truncate_singular_values(s, params)[0])
    A = to_lpr(ttn, f'qubit{b - 1}', N) if b >= 1 else None
    B = to_lpr(ttn, f'qubit{i + 1}', N) if i + 1 <= N - 1 else None
    left, right, _, _ = STATE['cut'](theta, k, 'L', A, B, b)
    STATE['cuts'] += 1
    left = left * nrm
    ttn.replace_tensor(next_id, from_lpr(ttn, next_id, left, N), new_shape=True)
    ttn.replace_tensor(node_id, from_lpr(ttn, node_id, right, N), new_shape=True)


def install(model, N, kappa, **kw):
    STATE.update(N=N, fired=0, cuts=0)
    STATE['cut'] = M.EnhCut(model, N, kappa=kappa, gamma=4.0, wE=kw.get('wE', 100.0),
                            ndir=kw.get('ndir', 2), window=kw.get('window', 'r2'),
                            energy_only=kw.get('energy_only', False))
    ST.split_svd_contract_sv_to_neighbour = _cut


def uninstall():
    ST.split_svd_contract_sv_to_neighbour = _ORIG
