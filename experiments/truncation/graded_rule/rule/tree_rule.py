"""Energy-aware subspace choice for the tree truncation walk.

At a child cut the orthogonality centre sits on the node, every other tensor is an isometry towards
it, and the child leg is cut by a rank-k projector Pi.  The represented state after the cut is
Pi T (T the node tensor), so the whole-chain energy is  E(Pi) = <Pi T|H_eff|Pi T> / <Pi T|Pi T>
with H_eff the single-site effective Hamiltonian of the node (TTNO plus environment blocks).

The rule moves the kept subspace off the SVD subspace U_k along the energy-error gradient,
Pi(t) = sum_j v_j v_j^dag,  v_j = cos(theta_j) u_j + sin(theta_j) q_j,  tan(theta_j) = t lam_j,
inside a trust region  ||T - Pi T||^2 <= (1 + kappa) * (SVD discarded weight).
Along this family E(t) is a quadratic form in 3k fixed tensors, so the line search costs 3k
effective-Hamiltonian applications per cut, independent of the grid.
"""
import numpy as np

import pytreenet.core.truncation.node_truncation as NT
from pytreenet.contractions.sandwich_caching import SandwichCache
from _mps_vendor.state_operator_contraction import contract_ket_ham_with_envs

STATE = dict(on=False, kappa=0.1, wE=1.0, ham=None, tree=None, fired=0, cuts=0, eps_floor=1e-14,
             ngrid=90)
_ORIG_CHILD = NT._child_projector
_ORIG_CUT = NT.node_cut_children


def configure(ham, kappa=0.1, wE=1.0, ngrid=90, ops=None):
    """ops: list of (ttno, weight); default is the Hamiltonian alone."""
    STATE.update(on=kappa > 0, kappa=kappa, wE=wE, ham=ham, fired=0, cuts=0, ngrid=ngrid,
                 ops=ops if ops is not None else [(ham, 1.0)])


def _cut_children(node_id, tree, svd_params, bond_params=None, preserve_legs_order=True, spectra=None):
    STATE['tree'] = tree
    return _ORIG_CUT(node_id, tree, svd_params, bond_params=bond_params,
                     preserve_legs_order=preserve_legs_order, spectra=spectra)


def install():
    NT._child_projector = _child_projector
    NT.node_cut_children = _cut_children


def uninstall():
    NT._child_projector = _ORIG_CHILD
    NT.node_cut_children = _ORIG_CUT


def _make_apply(tree, node, child_index, ham):
    cache = SandwichCache.init_cache_but_one(tree, ham, node.identifier)
    hnode, htensor = ham[node.identifier]
    shape = tree.tensors[node.identifier].shape
    rest = (shape[child_index],) + tuple(s for i, s in enumerate(shape) if i != child_index)

    def to_tensor(B):
        return np.moveaxis(B.reshape(rest), 0, child_index)

    def to_mat(T):
        return np.moveaxis(T, child_index, 0).reshape(shape[child_index], -1)

    def apply(B):
        return to_mat(contract_ket_ham_with_envs(node, to_tensor(B), hnode, htensor, cache))

    return apply


def energy_of_tree(tree, ham):
    """Whole-chain energy from the centre tensor, for validation."""
    node_id = tree.orthogonality_center_id
    node = tree.nodes[node_id]
    cache = SandwichCache.init_cache_but_one(tree, ham, node_id)
    hnode, htensor = ham[node_id]
    T = tree.tensors[node_id]
    HT = contract_ket_ham_with_envs(node, T, hnode, htensor, cache)
    return float(np.vdot(T, HT).real / np.vdot(T, T).real)


def _child_projector(node, node_tensor, child_id, svd_params):
    proj, sv = _ORIG_CHILD(node, node_tensor, child_id, svd_params)
    STATE['cuts'] += 1
    if not STATE['on']:
        return proj, sv
    k = proj.shape[1]
    ci = node.neighbour_index(child_id)
    M = np.moveaxis(node_tensor, ci, 0).reshape(node_tensor.shape[ci], -1)
    N = M.shape[0]
    if k >= N:
        STATE.setdefault('why',{}).setdefault('k>=N',0); STATE['why']['k>=N']+=1
        return proj, sv
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    tot = float(np.sum(s ** 2))
    eps = float(np.sum(s[k:] ** 2)) / tot
    if eps < STATE['eps_floor']:
        STATE.setdefault('why',{}).setdefault('eps',0); STATE['why']['eps']+=1
        return proj, sv
    out = _optimise(node, M, U, s, k, eps, ci, tot)
    if out is None:
        return proj, sv
    STATE['fired'] += 1
    return out, sv


def _optimise(node, M, U, s, k, eps, ci, tot):
    tree = STATE['tree']
    ops = STATE['ops']
    applies = [_make_apply(tree, node, ci, h) for h, _ in ops]
    ws = np.array([w for _, w in ops])
    Uk = U[:, :k]
    PM = Uk @ (Uk.conj().T @ M)
    den0 = float(np.vdot(PM, PM).real)
    gden = M @ (M.conj().T @ Uk)
    Eth = np.zeros(len(ops)); E0 = np.zeros(len(ops)); grads = []
    for a, ap in enumerate(applies):
        HM = ap(M)
        Eth[a] = float(np.vdot(M, HM).real) / tot
        HPM = ap(PM)
        E0[a] = float(np.vdot(PM, HPM).real) / den0
        G1 = HPM @ M.conj().T @ Uk
        G2 = M @ HPM.conj().T @ Uk
        grads.append((G1 + G2 - E0[a] * gden) / den0)
    dE0 = E0 - Eth
    f0 = float(np.sum(ws * dE0 ** 2))
    if f0 < 1e-30:
        STATE.setdefault('why', {}).setdefault('f0', 0); STATE['why']['f0'] += 1
        return None
    gsum = sum(2.0 * ws[a] * dE0[a] * grads[a] for a in range(len(ops)))
    Z0 = -(gsum - Uk @ (Uk.conj().T @ gsum))
    if float(np.linalg.norm(Z0)) < 1e-300:
        return None
    Qz, lam, Vzh = np.linalg.svd(Z0, full_matrices=False)
    lam = lam / lam[0]
    Uv = Uk @ Vzh.conj().T
    Bs = []
    for j in range(k):
        u, q = Uv[:, j:j + 1], Qz[:, j:j + 1]
        uM, qM = u.conj().T @ M, q.conj().T @ M
        Bs.append(u @ uM)
        Bs.append(u @ qM + q @ uM)
        Bs.append(q @ qM)
    n = len(Bs)
    Bst = np.array([B.reshape(-1) for B in Bs])
    S = (Bst.conj() @ Bst.T).real
    Ks = []
    for ap in applies:
        HB = np.array([ap(B).reshape(-1) for B in Bs])
        Ks.append((Bst.conj() @ HB.T).real)
    bud = (1.0 + STATE['kappa']) * eps * tot

    def evalt(t):
        th = np.arctan(t * lam)
        c = np.stack([np.cos(th) ** 2, np.cos(th) * np.sin(th), np.sin(th) ** 2], axis=1).reshape(-1)
        den = float(c @ S @ c)
        E = np.array([float(c @ K @ c) / den for K in Ks])
        return float(np.sum(ws * (E - Eth) ** 2)), tot - den

    best_t, best_f = 0.0, f0
    for t in np.logspace(-11, 3, STATE['ngrid']):
        f, d = evalt(t)
        if d <= bud and f < best_f:
            best_t, best_f = t, f
    if best_t > 0:
        for w in (0.5, 0.2, 0.08, 0.03):
            for t in best_t * np.linspace(1 - w, 1 + w, 13):
                f, d = evalt(t)
                if d <= bud and f < best_f:
                    best_t, best_f = t, f
    if best_t == 0.0 or best_f > f0 * (1 - 1e-9):
        STATE.setdefault('why', {}).setdefault('nobetter', 0); STATE['why']['nobetter'] += 1
        return None
    th = np.arctan(best_t * lam)
    return Uv * np.cos(th)[None, :] + Qz * np.sin(th)[None, :]
