"""Rank allocation under a parameter budget, solved exactly on the tree.

Minimises ``D(s) = sum_e tail_e(s_e)`` subject to ``P(s) = sum_v n_v prod_{e at v} s_e <= P_max``
via the Lagrangian ``L(s) = D(s) + mu * P(s)``. ``L`` is a sum of one term per bond and one per
node, so a leaves-to-root dynamic program finds its exact joint minimiser on a tree
(:func:`allocate`), and :func:`solve_budget` finds the multiplier for a given ``P_max`` by
walking the hull of achievable ``(P, D)`` points. Rank consistency (a bond carries no more than
``n_v`` times the product of the other legs at either endpoint) is enforced at every node.
"""
from dataclasses import dataclass
from typing import Dict, NamedTuple

import numpy as np

__all__ = ["ParameterBudgetWarning", "BudgetSolve", "Allocation", "Structure",
           "tail_table", "tables_from_spectra", "allocate", "solve_budget", "minimum_size"]

#: Relative tolerance on a tie in ``L``; the larger network wins, keeping ``P(mu)`` monotone.
TIE = 1e-12

#: Entries per chunk when a node's child-rank grid is evaluated. Memory ceiling only.
CHUNK = 2_000_000

#: Hull-edge steps allowed before the search returns what it has.
MAX_STEPS = 200


class ParameterBudgetWarning(UserWarning):
    """The target was out of reach of the bonds one truncation call was allowed to move."""


class BudgetSolve(NamedTuple):
    """What one budget solve settled on, for the caller to inspect after a step.

    Attributes:
        mu: The multiplier the allocation was taken at. ``0.0`` means already within target.
        realised: Parameters the cut left.
        per_bond: Parameters the per-bond criterion alone leaves (the measuring walk's size).
        trials: Truncation walks the solve spent, the final real cut excluded. Always one.
        feasible: False when even the smallest consistent network exceeds the target.
        solves: Dynamic programs evaluated to find the hull vertex. Cheap next to a walk.
    """
    mu: float
    realised: int
    per_bond: int
    trials: int
    feasible: bool = True
    solves: int = 0


@dataclass
class Allocation:
    """One exact minimiser of ``L`` and what it costs.

    Attributes:
        ranks: The rank kept on each bond, keyed by the bond's lower node.
        L: ``D + mu P`` at those ranks.
        P: Parameters the network stores at those ranks.
        D: Weight those ranks discard from the recorded singular values.
        mu: The multiplier this allocation minimises ``L`` at.
    """
    ranks: Dict[str, int]
    L: float
    P: int
    D: float
    mu: float


@dataclass
class Structure:
    """The rooted tree a budget is solved on, stripped of the tensors.

    A bond is named by its lower node, the convention every walk keys its recorded spectra by.

    Attributes:
        root: The root node's identifier.
        children: Each node's children, in leg order.
        phys: Each node's open-leg dimension, 1 for a node without one.
        fixed: Bonds the caller may not move, mapped to their dimension: counted in ``P``,
            never allocated.
    """
    root: str
    children: Dict[str, list]
    phys: Dict[str, int]
    fixed: Dict[str, int] = None

    def __post_init__(self):
        self.fixed = dict(self.fixed or {})
        self.parent = {self.root: None}
        order, stack = [], [(self.root, False)]
        while stack:
            v, done = stack.pop()
            if done:
                order.append(v)
                continue
            stack.append((v, True))
            for c in reversed(self.children[v]):
                self.parent[c] = v
                stack.append((c, False))
        self.post = order
        #: Every bond, and the subset that is actually allocated.
        self.all_bonds = [v for v in order if v != self.root]
        self.bonds = [e for e in self.all_bonds if e not in self.fixed]

    def legs(self, v):
        """The bonds at ``v``: its parent bond, named ``v``, and its child bonds."""
        return ([v] if v != self.root else []) + list(self.children[v])

    def params(self, s):
        """``P(s) = sum_v n_v prod_{e at v} s_e``, with the fixed bonds at their dimension."""
        full = {**self.fixed, **s}
        return int(sum(self.phys[v] * int(np.prod([full[e] for e in self.legs(v)]))
                       for v in self.children))

    def consistent(self, s):
        """Whether every bond is within what the legs at either endpoint can carry."""
        full = {**self.fixed, **s}
        for v in self.children:
            legs = self.legs(v)
            for e in legs:
                if full[e] > self.phys[v] * int(np.prod([full[f] for f in legs if f != e])):
                    return False
        return True

    @classmethod
    def from_ttn(cls, ttn, fixed=None):
        """Read the structure off a tree tensor network state."""
        children, phys = {}, {}
        for nid, node in ttn.nodes.items():
            children[nid] = list(node.children)
            phys[nid] = (int(np.prod([node.shape[leg] for leg in node.open_legs]))
                         if node.open_legs else 1)
        return cls(ttn.root_id, children, phys, fixed)


def tail_table(singular_values):
    """``t[s]`` = the weight discarded when ``s`` directions are kept, for ``s = 0 .. len``."""
    squared = np.asarray(singular_values, dtype=float) ** 2
    return np.concatenate([np.cumsum(squared[::-1])[::-1], [0.0]])


def tables_from_spectra(spectra):
    """``(tails, rmax)`` from a mapping of bond to the singular values a walk kept on it.

    ``rmax`` is the per-bond criterion's own kept count, the ceiling the budget allocates
    under: a budgeted cut never keeps a direction an unbudgeted one would have thrown away.
    """
    tails = {e: tail_table(sv) for e, sv in spectra.items()}
    rmax = {e: max(1, len(sv)) for e, sv in spectra.items()}
    return tails, rmax


def _ranges(structure, rmax, box):
    lo = {e: 1 for e in structure.bonds}
    hi = {e: int(rmax[e]) for e in structure.bonds}
    if box:
        for e, (a, b) in box.items():
            if e in lo:
                lo[e], hi[e] = max(lo[e], int(a)), min(hi[e], int(b))
    for e in structure.bonds:
        if lo[e] > hi[e]:
            raise ValueError(f"empty rank range on bond {e!r}: [{lo[e]}, {hi[e]}]")
    return lo, hi


def allocate(structure, tails, rmax, mu, box=None, tie=TIE, chunk=CHUNK):
    """The exact minimiser of ``L(s) = D(s) + mu P(s)`` over consistent ranks.

    Args:
        structure: The tree, from :meth:`Structure.from_ttn`.
        tails: Bond to its tail table, from :func:`tail_table`.
        rmax: Bond to the largest rank it may keep, from :func:`tables_from_spectra`.
        mu: The multiplier. ``0`` returns the per-bond criterion's own ranks.
        box: Optional bond to inclusive rank interval, a trust region. Bonds left out keep
            their full range.
        tie: Relative tolerance on a tie in ``L``; the larger network wins a tie.
        chunk: Memory ceiling for a node's grid. Does not change the result.

    Returns:
        The :class:`Allocation`.

    Raises:
        ValueError: No consistent assignment exists in the given ranges.
    """
    lo, hi = _ranges(structure, rmax, box)
    fixed = structure.fixed

    def rank_axis(e):
        """The values bond ``e`` may take: one if it is held, its range otherwise."""
        if e in fixed:
            return np.array([float(fixed[e])])
        return np.arange(lo[e], hi[e] + 1, dtype=float)

    M, PM, ARG, SHAPE = {}, {}, {}, {}
    for v in structure.post:
        kids = structure.children[v]
        n = float(structure.phys[v])
        xs = np.array([1.0]) if v == structure.root else rank_axis(v)
        if not kids:
            m = mu * n * xs
            pm = n * xs
            m[xs > n] = np.inf
            M[v], PM[v] = m, pm
            continue
        ys = [rank_axis(c) for c in kids]
        gL, gP = [], []
        for c in kids:
            if c in fixed:
                gL.append(np.array([tails[c][fixed[c]] if c in tails else 0.0]) + M[c])
            else:
                gL.append(tails[c][lo[c]:hi[c] + 1] + M[c])
            gP.append(PM[c])
        G, GP, prod = gL[0], gP[0], ys[0]
        for i in range(1, len(kids)):
            G = np.add.outer(G, gL[i])
            GP = np.add.outer(GP, gP[i])
            prod = np.multiply.outer(prod, ys[i])
        shape = G.shape if len(kids) > 1 else (len(ys[0]),)
        G, GP, prod = G.ravel(), GP.ravel(), prod.ravel()
        yflat = [np.broadcast_to(
            ys[i].reshape([-1 if j == i else 1 for j in range(len(kids))]), shape).ravel()
            for i in range(len(kids))]
        m = np.empty(len(xs))
        pm = np.empty(len(xs))
        arg = np.full(len(xs), -1, dtype=np.int64)
        step = max(1, chunk // max(1, G.size))
        for a in range(0, len(xs), step):
            x = xs[a:a + step, None]
            L = mu * n * x * prod[None, :] + G[None, :]
            P = n * x * prod[None, :] + GP[None, :]
            valid = np.isfinite(L)
            if v != structure.root:
                valid &= x <= n * prod[None, :]
            for yi in yflat:
                valid &= (yi[None, :] ** 2) <= n * x * prod[None, :]
            L = np.where(valid, L, np.inf)
            best = L.min(axis=1)
            cand = L <= best[:, None] * (1.0 + tie) + 1e-300  # ties: larger network wins
            pick = np.argmax(np.where(cand, P, -np.inf), axis=1)
            rows = np.arange(L.shape[0])
            m[a:a + step] = L[rows, pick]
            pm[a:a + step] = P[rows, pick]
            arg[a:a + step] = np.where(np.isfinite(best), pick, -1)
        M[v], PM[v], ARG[v], SHAPE[v] = m, pm, arg, shape

    root = structure.root
    if not np.isfinite(M[root][0]):
        raise ValueError("no consistent rank assignment exists in the given ranges")

    ranks = {}

    def assign(v, xi):
        kids = structure.children[v]
        if not kids:
            return
        idx = np.unravel_index(int(ARG[v][xi]), SHAPE[v])
        for c, j in zip(kids, idx):
            ranks[c] = int(fixed[c]) if c in fixed else int(lo[c] + j)
            assign(c, int(j))

    assign(root, 0)
    allocated = {e: ranks[e] for e in structure.bonds}
    D = float(sum(tails[e][allocated[e]] for e in structure.bonds))
    P = structure.params(allocated)
    return Allocation(allocated, D + mu * P, P, D, float(mu))


def _mu_above_everything(structure, tails, lo):
    """A multiplier at which the smallest consistent network is optimal, i.e. ``D(lo) < mu``."""
    return 2.0 * float(sum(tails[e][lo[e]] for e in structure.bonds)) + 1e-300


def solve_budget(structure, tails, rmax, p_max, box=None):
    """The hull vertex with the largest ``P <= p_max``, by hull-edge steps.

    Each step solves at the slope ``mu = (D_b - D_a) / (P_a - P_b)`` of the edge between the
    current infeasible point ``a`` and feasible point ``b``; a point strictly below that edge
    replaces one end, and the loop ends when ``a`` and ``b`` are adjacent hull vertices.

    Returns:
        ``(allocation, feasible, solves)``. ``feasible`` is False when even the smallest
        consistent network exceeds ``p_max``, in which case that network is returned.
    """
    lo, _ = _ranges(structure, rmax, box)
    a = allocate(structure, tails, rmax, 0.0, box)
    solves = 1
    if a.P <= p_max:
        return a, True, solves
    b = allocate(structure, tails, rmax, _mu_above_everything(structure, tails, lo), box)
    solves += 1
    if b.P > p_max:
        return b, False, solves
    for _ in range(MAX_STEPS):
        if a.P <= b.P:
            break
        mu = max((b.D - a.D) / (a.P - b.P), 1e-300)
        s = allocate(structure, tails, rmax, mu, box)
        solves += 1
        la, lb = a.D + mu * a.P, b.D + mu * b.P
        if s.L >= min(la, lb) * (1.0 - 1e-10) or s.ranks in (a.ranks, b.ranks):
            break
        if s.P <= p_max:
            b = s
        else:
            a = s
    return b, True, solves


def minimum_size(ttn) -> int:
    """Parameters this network holds with every bond at dimension one, its floor."""
    total = 0
    for node_id, tensor in ttn.tensors.items():
        node = ttn.nodes[node_id]
        open_size = tensor.size
        for neighbour_id in node.neighbouring_nodes():
            open_size //= tensor.shape[node.neighbour_index(neighbour_id)]
        total += open_size
    return total
