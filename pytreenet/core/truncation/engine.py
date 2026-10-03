"""Post-sweep truncation: the walk enum and the engine that runs it."""
from copy import deepcopy
from dataclasses import fields
from enum import Enum
from typing import List, Optional
from warnings import catch_warnings, filterwarnings, warn

from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.util.tensor_splitting import SVDParameters

from .parameter_budget import (BudgetSolve, ParameterBudgetWarning, Structure,
                               minimum_size, solve_budget, tables_from_spectra)
from .node_truncation import recursive_node_cut_truncation
from .sweeping_truncation import sweeping_per_bond_truncation
from .truncation_util import find_orthogonalization_path

__all__ = ["TruncationWalk", "TruncationEngine"]


class TruncationWalk(Enum):
    """How the truncation traverses the network. Set by the caller, never by a config.

    Attributes:
        SWEEPING: Follow a path, leaving the centre at its last node. Can be restricted to
            part of the path.
        RECURSIVE: Post-order DFS of node cuts over the whole network, ending at the root,
            each bond cut once at the turn of its upper endpoint. Rejects a path.
    """
    SWEEPING = "sweeping"
    RECURSIVE = "recursive"


class TruncationEngine:
    """Truncates a tree tensor network state by an SVD cut on every bond.

    Build it with :meth:`sweeping` or :meth:`recursive`. If the config passed to
    :meth:`truncate` has a ``parameter_budget``, the rank cap is replaced by the exact
    allocation of :mod:`~pytreenet.core.truncation.parameter_budget`.

    Args:
        walk: How to traverse the network.
        initial_ttn: The state whose topology fixes the paths. Required by a sweeping
            engine, unused by a recursive one.
        forward_trunc_path: The forward leaf-to-leaf path. Required by a sweeping engine.
        backward_trunc_path: The backward leaf-to-leaf path. Required by a sweeping engine.

    Raises:
        TypeError: ``walk`` is not a TruncationWalk.
        ValueError: A sweeping engine was built without the state and both paths.
    """

    def __init__(self,
                 walk: TruncationWalk,
                 initial_ttn: Optional[TreeTensorNetworkState] = None,
                 forward_trunc_path: Optional[List[str]] = None,
                 backward_trunc_path: Optional[List[str]] = None):
        if not isinstance(walk, TruncationWalk):
            raise TypeError(
                f"truncation walk must be a TruncationWalk, not {type(walk).__name__}. "
                f"Valid walks: {', '.join(w.name for w in TruncationWalk)}.")
        self.walk = walk
        #: Orthogonalisation paths for caller-supplied sub-paths, keyed by the sub-path.
        self._orth_path_cache = {}
        #: What the most recent budgeted call settled on, or None if it had no budget.
        self.last_budget_solve = None
        self._budget_warned = False
        if walk is TruncationWalk.RECURSIVE:
            return
        if initial_ttn is None or forward_trunc_path is None or backward_trunc_path is None:
            raise ValueError("a sweeping engine needs the state and both leaf-to-leaf "
                             "paths; pass them to TruncationEngine.sweeping.")
        self.forward_trunc_path = forward_trunc_path
        self.forward_orth_path = find_orthogonalization_path(
            initial_ttn, self.forward_trunc_path)
        self.backward_trunc_path = backward_trunc_path
        self.backward_orth_path = find_orthogonalization_path(
            initial_ttn, self.backward_trunc_path)

    @classmethod
    def sweeping(cls, initial_ttn: TreeTensorNetworkState,
                forward_trunc_path: List[str],
                backward_trunc_path: List[str]) -> "TruncationEngine":
        """An engine that follows the two given leaf-to-leaf paths, or a part of one."""
        return cls(TruncationWalk.SWEEPING, initial_ttn,
                  forward_trunc_path, backward_trunc_path)

    @classmethod
    def recursive(cls) -> "TruncationEngine":
        """An engine that recurses from the root over the whole network."""
        return cls(TruncationWalk.RECURSIVE)

    def truncate(self,
                 ttn: TreeTensorNetworkState,
                 svd_params: SVDParameters,
                 forward: bool = True,
                 path: Optional[list] = None,
                 preserve_legs_order: bool = False) -> None:
        """Truncate ``ttn`` in place, by this engine's walk.

        Args:
            ttn: The state to truncate.
            svd_params: SVD parameters. ``max_bond_dim`` caps every bond.
            forward: Sweep direction. Read by the sweeping walk only.
            path: Compress only this path. Supersedes ``forward``; sweeping walk only.
            preserve_legs_order: Keep each node's neighbour order. The recursive walk
                always does.

        Raises:
            ValueError: A ``path`` was given to a recursive engine, or the budget is below
                the network's size at bond dimension one.
        """
        if self.walk is TruncationWalk.RECURSIVE and path is not None:
            raise ValueError(
                "a recursive truncation covers the whole network from the root and "
                "cannot be restricted to a path. Build a sweeping engine to compress "
                "a window.")
        target = getattr(svd_params, "parameter_budget", None)
        if target is None:
            self.last_budget_solve = None
            self._walk(ttn, svd_params, forward, path, preserve_legs_order)
            return
        self._truncate_to_budget(ttn, svd_params, forward, path, preserve_legs_order,
                                 int(target))

    def _walk(self,
              ttn: TreeTensorNetworkState,
              svd_params: SVDParameters,
              forward: bool,
              path: Optional[list],
              preserve_legs_order: bool,
              bond_params=None,
              spectra=None) -> None:
        """One truncation walk. ``bond_params`` gives each bond its own cut; ``spectra``, if
        given, is filled with every cut bond's kept singular values."""
        if self.walk is TruncationWalk.RECURSIVE:
            recursive_node_cut_truncation(ttn, svd_params, bond_params=bond_params,
                                          spectra=spectra)
            return

        if path is None:
            trunc_path = (self.forward_trunc_path if forward
                          else self.backward_trunc_path)
            orth_path = (self.forward_orth_path if forward
                         else self.backward_orth_path)
        else:
            trunc_path, orth_path = path, self._orth_path_for(ttn, path)
        sweeping_per_bond_truncation(ttn, trunc_path, orth_path, svd_params,
                                     preserve_legs_order=preserve_legs_order,
                                     bond_params=bond_params, spectra=spectra)

    def _truncate_to_budget(self,
                            ttn: TreeTensorNetworkState,
                            svd_params: SVDParameters,
                            forward: bool,
                            path: Optional[list],
                            preserve_legs_order: bool,
                            target: int) -> None:
        """Measure the spectra in one walk, solve the allocation, cut once at it.

        Raises:
            ValueError: The target is below the bond-one floor, or ``renorm`` is set.
        """
        if getattr(svd_params, "renorm", False):
            raise ValueError(
                "parameter_budget cannot be used with renorm=True: the allocation reads the "
                "singular values the measuring walk keeps as the state's Schmidt "
                "coefficients, and renormalization rescales them.")
        floor = minimum_size(ttn)
        if target < floor:
            raise ValueError(
                f"parameter_budget={target} is below the {floor} parameters this network "
                "holds with every bond at dimension one, so no truncation can reach it. "
                "Reduce the physical dimensions or the number of sites.")

        spectra = {}
        trial = deepcopy(ttn)
        with catch_warnings():
            # A bond entirely under the tolerance cuts to rank one; harmless here.
            filterwarnings("ignore", message="All singular values were truncated")
            self._walk(trial, svd_params, forward, path, preserve_legs_order,
                       spectra=spectra)
        per_bond = trial.size()

        # Bonds outside this call's walk keep their dimension: counted in P, never allocated.
        fixed = {e: int(ttn.nodes[e].parent_leg_dim())
                 for e in self._bonds_outside(ttn, spectra)}
        structure = Structure.from_ttn(ttn, fixed=fixed)
        tails, rmax = tables_from_spectra(spectra)
        allocation, feasible, solves = solve_budget(structure, tails, rmax, target)

        if not feasible and not self._budget_warned:
            self._budget_warned = True
            warn(f"parameter_budget={target} is not reachable with the bonds this "
                 "truncation may move; cutting to the smallest network it can reach. An "
                 "untouched part of the network already exceeds the budget.",
                 ParameterBudgetWarning, stacklevel=3)

        # Only the cap and ``random`` change, so a slack budget reproduces the unbudgeted walk.
        base = {f.name: getattr(svd_params, f.name, f.default) for f in fields(SVDParameters)}
        params = {e: SVDParameters(**{**base, "max_bond_dim": int(r), "random": False})
                  for e, r in allocation.ranks.items()}

        def bond_params(tree, node_id, neighbour_id):
            """The cut for one bond, named by its lower node whatever the call order."""
            bond = (neighbour_id if tree.nodes[neighbour_id].parent == node_id else node_id)
            return params.get(bond, svd_params)

        with catch_warnings():
            filterwarnings("ignore", message="All singular values were truncated")
            self._walk(ttn, svd_params, forward, path, preserve_legs_order, bond_params)
        self.last_budget_solve = BudgetSolve(allocation.mu, ttn.size(), per_bond,
                                             trials=1, feasible=feasible, solves=solves)

    @staticmethod
    def _bonds_outside(ttn, spectra) -> list:
        """Bonds the measuring walk did not reach, which this call may therefore not move."""
        return [nid for nid in ttn.nodes
                if nid != ttn.root_id and nid not in spectra]

    def _orth_path_for(self, ttn: TreeTensorNetworkState, path: list) -> list:
        """The orthogonalisation path for a caller-supplied ``path``, memoised."""
        key = tuple(path)
        orth_path = self._orth_path_cache.get(key)
        if orth_path is None:
            orth_path = find_orthogonalization_path(ttn, path)
            self._orth_path_cache[key] = orth_path
        return orth_path
