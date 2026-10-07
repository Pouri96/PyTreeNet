"""Product MPS on a chain, built from pytreenet alone (the one helper the RAGE bench takes from the RAGE benchmarks)."""
from __future__ import annotations

import _pytreenet_root  # noqa: F401

import pytreenet as ptn


def product_mps(local_indices, phys_dim, bond_dim, node_prefix="qubit", root_site=0):
    """Padded product MPS |i_0, i_1, ...> with per-site basis indices."""
    n = len(local_indices)
    state = ptn.MatrixProductState.constant_product_state(
        0, phys_dim, n, bond_dimensions=[bond_dim] * (n - 1),
        node_prefix=node_prefix, root_site=root_site)
    for site, idx in enumerate(local_indices):
        t = state.tensors[f"{node_prefix}{site}"]
        t[...] = 0.0
        t[(0,) * (t.ndim - 1) + (int(idx),)] = 1.0
    return state
