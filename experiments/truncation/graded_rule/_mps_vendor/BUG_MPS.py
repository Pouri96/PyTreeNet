"""BUG-MPS: the Basis-Update and Galerkin integrator for matrix product states.

One time step is a single sweep along the chain. At each site the tensor is evolved (the
K-step), the ``[old | evolved]`` pair is QR-ed into a new basis, and the basis change is
absorbed into the next site; the chain end carries the Galerkin solve. Bonds are truncated
afterwards, so the rank at most doubles during the sweep.

``config.warmup_sweeps`` adds basis-only sweeps before a step (see ``WarmupConfig``).

Site tensors carry legs ``(D_left, D_right, d)``, with the boundary legs absent at the two
ends.

Example
-------
>>> config = BUGMPSConfig(max_bond_dim=64, rel_tol=1e-10)
>>> solver = BUG_MPS(state, hamiltonian, 0.01, 1.0, observables, config)
>>> solver.run()
"""

from typing import Dict, List, Union

import numpy as np
from numpy.linalg import qr

from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.state_operator_contraction import (contract_leaf,
                                                               contract_any)
from pytreenet.core.truncation import TruncationEngine
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.ttno.time_dep_ttno import AbstractTimeDepTTNO

from .ttn_time_evolution import TTNTimeEvolution
from .bug_util import BUGMPSConfig, SweepComposition, warmup_passes
from .common_bug import warn_nonhermitian_krylov, validate_config
from .effective_time_evolution import single_site_time_evolution
from .update_path import SweepingUpdatePathFinder, PathFinderMode

__all__ = ["BUG_MPS"]

Observables = Union[List[Union[TensorProduct, TreeTensorNetworkOperator]],
                    Dict[str, Union[TensorProduct, TreeTensorNetworkOperator]],
                    TensorProduct,
                    TreeTensorNetworkOperator]


class BUG_MPS(TTNTimeEvolution):
    """Basis-Update and Galerkin integrator for matrix product states.

    See the module docstring for the sweep and the leg conventions.

    Args:
        initial_state: The MPS to evolve. Copied by the base class.
        hamiltonian: The Hamiltonian as an MPO. Must be an
            AbstractTimeDepTTNO when ``config.time_dep`` is set.
        time_step_size: Step size ``dt``.
        final_time: Time at which the evolution stops.
        operators: Observables to measure during the run.
        config: A BUGMPSConfig; defaults are used when omitted.

    Attributes:
        state: The current MPS.
        hamiltonian: The Hamiltonian MPO.
        update_path: Site ids in sweep order (left to right).
        partial_tree_cache: Environment blocks ``<bra|H|ket>`` toward the site
            currently being updated.
        trunc_engine: Performs the post-sweep truncation, sweeping the chain per bond.
    """
    config_class = BUGMPSConfig

    def __init__(self,
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TreeTensorNetworkOperator,
                 time_step_size: float,
                 final_time: float,
                 operators: Observables,
                 config: Union[BUGMPSConfig, None] = None) -> None:
        super().__init__(initial_state, time_step_size, final_time, operators, config)

        if self.config.time_dep and not isinstance(hamiltonian, AbstractTimeDepTTNO):
            raise TypeError("The Hamiltonian must be from the AbstractTimeDepTTNO class "
                            "if the time evolution is time-dependent!")
        self.hamiltonian = hamiltonian
        if self.config.time_dep:
            # Rewind the TTNO's layer cursor, so a TTNO reused across solvers starts
            # this run at its first layer.
            self.reset_hamiltonian()
        self.state: TreeTensorNetworkState
        self.config: BUGMPSConfig

        self.update_path = SweepingUpdatePathFinder(
            self.initial_state, PathFinderMode.LeafToLeaf_Backward).find_path()
        #: The two end-to-end traversals, keyed by whether they are the forward one.
        self._endpoint_paths = {
            False: self.update_path,
            True: SweepingUpdatePathFinder(
                self.initial_state, PathFinderMode.LeafToLeaf_Forward).find_path(),
        }
        first_site = self.update_path[0]
        # Left-canonical with the centre at the sweep start. The cache then holds a
        # block for every other site.
        if self.state.orthogonality_center_id is None:
            self.state.canonical_form(first_site, preserve_legs_order=True)
        else:
            self.state.move_orthogonalization_center(first_site,
                                                     preserve_legs_order=True)
        self.partial_tree_cache = SandwichCache.init_cache_but_one(
            self.state, self.hamiltonian, first_site)

        # The sweep leaves the centre at the far end, so truncation follows the same path.
        self.trunc_engine = TruncationEngine.sweeping(
            self.initial_state, self._endpoint_paths[True], self._endpoint_paths[False])
        self._steps_done = 0            # completed steps; gates the warm-up

        validate_config(self)
        warn_nonhermitian_krylov(self)

    # ------------------------------------------------------- basis updates
    def _update_first_site(self, site_id: str, right_neighbor_id: str) -> None:
        """Basis update at the root: K-step, then the thin QR of ``[old | evolved]``.

        The root has no parent and one child, so its tensor is ``(D_right, d)``.
        """
        # Order matters: the K-step reads state[site_id] and must finish before the
        # centre move mutates the state. Evolving afterwards would read a moved centre.
        mode, options = self.config.local_solver()
        evolved_tensor = single_site_time_evolution(
            site_id, self.state, self.hamiltonian, self.time_step_size,
            self.partial_tree_cache, mode=mode, solver_options=options)
        self.state.move_orthogonalization_center(right_neighbor_id,
                                                 preserve_legs_order=True)
        old = self.state.tensors[site_id]                    # (D_right, d)
        combined = np.concatenate([old, evolved_tensor], axis=0)
        new_basis_matrix, _ = qr(combined.T)                 # (d, <= 2 D_right)
        new_basis_tensor = new_basis_matrix.T                # (k, d)
        self._install_basis(site_id, right_neighbor_id, old, new_basis_tensor)
        state_node, state_tensor = self.state[site_id]
        op_node, op_tensor = self.hamiltonian[site_id]
        env = contract_leaf(state_node, state_tensor, op_node, op_tensor)
        self.partial_tree_cache.add_entry(site_id, right_neighbor_id, env)

    def _update_middle_site(self, site_id: str, next_id: str) -> None:
        """Basis update at an interior site: K-step, then thin QR of ``[old | evolved]``.

        The leg facing ``next_id`` depends on the sweep direction, so it is read from the node.
        """
        # Order matters -- see _update_first_site.
        mode, options = self.config.local_solver()
        evolved_tensor = single_site_time_evolution(
            site_id, self.state, self.hamiltonian, self.time_step_size,
            self.partial_tree_cache, mode=mode, solver_options=options)
        self.state.move_orthogonalization_center(next_id, preserve_legs_order=True)
        old = self.state.tensors[site_id]                    # (D_parent, D_child, d)
        bond_leg = self.state.nodes[site_id].neighbour_index(next_id)
        kept = [leg for leg in range(old.ndim) if leg != bond_leg]
        kept_shape = tuple(old.shape[leg] for leg in kept)
        perm = (*kept, bond_leg)
        combined = np.concatenate([old, evolved_tensor], axis=bond_leg)
        matrix_form = np.transpose(combined, perm).reshape(int(np.prod(kept_shape)), -1)
        new_basis_matrix, _ = qr(matrix_form)                # (prod(kept), k)
        k = new_basis_matrix.shape[1]
        new_basis_tensor = np.transpose(
            new_basis_matrix.reshape(*kept_shape, k), np.argsort(perm))
        self._install_basis(site_id, next_id, old, new_basis_tensor)
        env = contract_any(node_id=site_id, next_node_id=next_id,
                           state=self.state, operator=self.hamiltonian,
                           dictionary=self.partial_tree_cache)
        self.partial_tree_cache.add_entry(site_id, next_id, env)

    def _install_basis(self, site_id: str, right_neighbor_id: str, old: np.ndarray,
                       new_basis_tensor: np.ndarray) -> None:
        """Install ``new_basis_tensor`` at ``site_id`` and push the basis change right.

        The basis-change matrix is ``M = old^dagger new``, contracted over every leg
        except the one facing ``right_neighbor_id``. Absorbing it into the neighbour
        leaves the represented state unchanged; only the bond basis widens.
        """
        bond_leg = self.state.nodes[site_id].neighbour_index(right_neighbor_id)
        legs = [leg for leg in range(new_basis_tensor.ndim) if leg != bond_leg]
        basis_change = np.tensordot(old, new_basis_tensor.conj(), axes=(legs, legs))
        self.state.absorb_matrix_into_neighbour_leg(right_neighbor_id, site_id,
                                                    basis_change, tensor_leg=0)
        self.state.replace_tensor(site_id, new_basis_tensor, new_shape=True)

    def _update_last_site(self, site_id: str) -> None:
        """Galerkin step at the chain end: evolve in place.

        The last site has no right neighbour, so there is no basis to expand and no
        basis change to propagate; the evolved tensor is the updated site tensor.
        """
        mode, options = self.config.local_solver()
        evolved_tensor = single_site_time_evolution(
            site_id, self.state, self.hamiltonian, self.time_step_size,
            self.partial_tree_cache, mode=mode, solver_options=options)
        self.state.replace_tensor(site_id, evolved_tensor, new_shape=True)

    # ------------------------------------------------------------ time step
    def sweeping_update(self, galerkin: bool = True) -> None:
        """Run one left-to-right sweep: basis update at every site, Galerkin at the last.

        Args:
            galerkin: Run the chain-end Galerkin solve. ``False`` makes the sweep a
                basis-only pass: every bond widens and the state is unchanged.
        """
        path = self.update_path
        last = len(path) - 1
        for i, site_id in enumerate(path):
            if i == 0:
                self._update_first_site(site_id, path[1])
            elif i == last:
                if galerkin:
                    self._update_last_site(site_id)
            else:
                self._update_middle_site(site_id, path[i + 1])

    def prepare_next_sweep(self) -> None:
        """Re-gauge to the first site and rebuild the environment cache."""
        first_site = self.update_path[0]
        if self.state.orthogonality_center_id:
            self.state.move_orthogonalization_center(first_site, preserve_legs_order=True)
        else:
            self.state.canonical_form(first_site)
        self.partial_tree_cache = SandwichCache.init_cache_but_one(
            self.state, self.hamiltonian, first_site)

    def _use_endpoint(self, forward: bool) -> None:
        """Point the sweep at one of the two end-to-end traversals."""
        self.update_path = self._endpoint_paths[forward]

    def _half_sweep(self, forward: bool, prepare: bool = True) -> None:
        """One sweep from the given endpoint, then compression along the reverse path.

        ``prepare`` is False for the first half of a step, already prepared by the last.
        """
        self._use_endpoint(forward)
        if prepare:
            # The state moved in the previous half, so the environment cache cannot be
            # carried across; this re-gauges to the new path's start and rebuilds it.
            self.prepare_next_sweep()
        self.sweeping_update()
        self.trunc_engine.truncate(self.state, self.config, forward=not forward)

    def _warmup(self) -> None:
        """Take this step's basis-only sweeps (once per step, not per half-sweep), each
        followed by a re-gauge to the forward start."""
        for _ in range(warmup_passes(self.config, self._steps_done)):
            self._use_endpoint(False)
            self.sweeping_update(galerkin=False)
            self.prepare_next_sweep()

    def run_one_time_step(self) -> None:
        """Advance the state by ``dt``, by this config's :class:`SweepComposition`."""
        self._warmup()
        self._steps_done += 1
        if self.config.composition is SweepComposition.SINGLE:
            self._use_endpoint(False)
            self.sweeping_update()
            self.trunc_engine.truncate(self.state, self.config, forward=True)
            self.prepare_next_sweep()
            return

        # Two half-sweeps of dt/2 from opposite ends, each compressed.
        full_step = self._time_step_size
        self._time_step_size = full_step / 2
        try:
            # The step is entered gauged at the first path's start with a live cache,
            # left there by __init__ or by the trailing call below.
            self._half_sweep(False, prepare=False)
            self._half_sweep(True)
        finally:
            self._time_step_size = full_step
        self._use_endpoint(False)
        self.prepare_next_sweep()

    # ------------------------------------------------- time-dependent hooks
    def update_hamiltonian(self) -> None:
        """Advance a time-dependent Hamiltonian, apply the layer's state change, and rebuild
        the environment cache. Called by ``run`` only when ``config.time_dep`` is set."""
        self.hamiltonian.update(self.time_step_size)
        self.hamiltonian.modify_state(self.state)
        self.prepare_next_sweep()
