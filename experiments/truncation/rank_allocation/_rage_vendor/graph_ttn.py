"""Build a product tree tensor network state on any tree given as an edge list.

Every node carries one open leg (dimension ``phys_dim``, or 1 for a virtual node via
``phys_dims``).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pytreenet.core.node import Node
from pytreenet.ttns.ttns import TreeTensorNetworkState

__all__ = ["tree_adjacency", "rooted_tree_order", "ttns_from_tree"]

Label = Hashable
Edge = Tuple[Label, Label]


def tree_adjacency(tree_edges: Iterable[Edge]) -> Dict[Label, List[Label]]:
    """Adjacency lists of an undirected edge list, each neighbour list in insertion order.

    Raises:
        ValueError: The edge list contains a self-loop or a duplicate edge.
    """
    adj: Dict[Label, List[Label]] = defaultdict(list)
    seen = set()
    for u, v in tree_edges:
        if u == v:
            raise ValueError(f"self-loop at {u!r}: a tree has no self-loops.")
        key = frozenset((u, v))
        if key in seen:
            raise ValueError(f"duplicate edge {(u, v)!r}.")
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    return dict(adj)


def rooted_tree_order(tree_edges: Iterable[Edge], root: Label
                      ) -> Tuple[List[Label], Dict[Label, Optional[Label]],
                                 Dict[Label, List[Label]]]:
    """Root the edge list at ``root``, returning ``(bfs_order, parent, children)``, after
    checking that it is a connected, acyclic tree.

    Raises:
        ValueError: ``root`` is absent from a non-empty edge list, or the edge list is
            cyclic or disconnected. An empty edge list returns the lone ``root``.
    """
    adj = tree_adjacency(tree_edges)
    if root not in adj:
        if not adj:
            return [root], {root: None}, {root: []}    # a lone node is a degenerate tree
        raise ValueError(f"root {root!r} does not appear in the edge list.")
    n_edges = sum(len(nb) for nb in adj.values()) // 2
    parent: Dict[Label, Optional[Label]] = {root: None}
    children: Dict[Label, List[Label]] = {root: []}
    order: List[Label] = [root]
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        for nxt in adj[cur]:
            if nxt == parent[cur]:
                continue
            if nxt in parent:
                raise ValueError(
                    f"the edge list contains a cycle (reached {nxt!r} twice); a tree "
                    f"tensor network needs an acyclic topology. For a device graph with "
                    f"loops, pass a spanning tree and route the loop-closing gates as "
                    f"long windows.")
            parent[nxt] = cur
            children[nxt] = []
            children[cur].append(nxt)
            order.append(nxt)
            queue.append(nxt)
    if len(order) != len(adj):
        unreached = sorted(set(adj) - set(order), key=repr)[:5]
        raise ValueError(
            f"the edge list is disconnected; {unreached} unreachable from {root!r}.")
    if n_edges != len(order) - 1:
        raise ValueError(
            f"expected {len(order) - 1} edges for {len(order)} nodes, got {n_edges}.")
    return order, parent, children


def _node_tensor(n_children: int, is_root: bool, phys_vector: np.ndarray,
                 bond_dim: int, dtype) -> np.ndarray:
    """Product-state tensor with bonds of ``bond_dim``; only the ``(0, ..., 0)`` bond entry
    is non-zero, so the state is the same at any ``bond_dim``."""
    shape = (bond_dim,) * (n_children + (0 if is_root else 1)) + (len(phys_vector),)
    tensor = np.zeros(shape, dtype=dtype)
    tensor[(0,) * (len(shape) - 1)] = phys_vector
    return tensor


def ttns_from_tree(tree_edges: Iterable[Edge],
                   root: Label,
                   local_kets: Optional[Dict[Label, Sequence[complex]]] = None,
                   node_ids: Optional[Dict[Label, str]] = None,
                   node_prefix: str = "q",
                   phys_dim: int = 2,
                   bond_dim: int = 1,
                   dtype=complex,
                   phys_dims: Optional[Dict[Label, int]] = None
                   ) -> Tuple[TreeTensorNetworkState, Dict[Label, str]]:
    """A product state on the tree given by ``tree_edges``, with one open leg per node.

    Args:
        tree_edges: Undirected edges ``(u, v)`` over hashable labels. Must form a tree.
        root: The label to use as the network root.
        local_kets: ``{label: vector}``, each of that label's physical dimension.
            Unlisted labels get ``|0>``. Vectors are normalised.
        node_ids: ``{label: identifier}``; defaults to ``f"{node_prefix}{label}"``.
        node_prefix: Prefix for the default identifiers.
        phys_dim: Default physical leg dimension -- 2 for a qubit.
        bond_dim: Initial dimension of every bond (zero-padded above 1).
        dtype: Tensor dtype. Pass ``float`` for a real network.
        phys_dims: Per-label override of ``phys_dim``, e.g. 1 for a virtual node.

    Returns:
        ``(state, ids)``: the network and the ``{label: identifier}`` map.

    Raises:
        ValueError: The topology is not a tree, ``node_ids`` is incomplete or not
            injective, ``bond_dim`` is below 1, or a local ket has the wrong length.
    """
    order, parent, children = rooted_tree_order(tree_edges, root)
    ids = (dict(node_ids) if node_ids is not None
           else {lab: f"{node_prefix}{lab}" for lab in order})
    missing = [lab for lab in order if lab not in ids]
    if missing:
        raise ValueError(f"node_ids has no identifier for {missing[:5]}.")
    if len(set(ids[lab] for lab in order)) != len(order):
        raise ValueError("node_ids must be injective; two labels share an identifier.")
    if bond_dim < 1:
        raise ValueError(f"bond_dim must be >= 1, got {bond_dim}.")

    def dim_of(lab: Label) -> int:
        return int(phys_dims.get(lab, phys_dim)) if phys_dims is not None else phys_dim

    def ket(lab: Label) -> np.ndarray:
        d = dim_of(lab)
        if local_kets is None or lab not in local_kets:
            vec = np.zeros(d, dtype=dtype)
            vec[0] = 1.0
            return vec
        vec = np.asarray(local_kets[lab], dtype=dtype).reshape(-1)
        if vec.size != d:
            raise ValueError(f"local ket at {lab!r} has length {vec.size}, expected {d}.")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    state = TreeTensorNetworkState()
    state.add_root(Node(identifier=ids[root]),
                   _node_tensor(len(children[root]), True, ket(root), bond_dim, dtype))
    for lab in order[1:]:
        parent_id = ids[parent[lab]]
        parent_node = state.nodes[parent_id]
        # Next free open leg, read off the live node: attaching a child consumes one.
        parent_leg = parent_node.nparents() + parent_node.nchildren()
        state.add_child_to_parent(
            Node(identifier=ids[lab]),
            _node_tensor(len(children[lab]), False, ket(lab), bond_dim, dtype),
            0, parent_id, parent_leg)
    return state, ids
