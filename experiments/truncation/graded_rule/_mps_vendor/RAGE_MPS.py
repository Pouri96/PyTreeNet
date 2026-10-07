"""RAGE-MPS: Rank-Adaptive Galerkin Evolution for matrix product states.

Keeps the sweep structure of BUG_MPS but replaces how each bond basis is enlarged. Where
BUG_MPS evolves the site and keeps ``[old | evolved]``, doubling every rank, RAGE-MPS keeps
the floor ``QR(old)`` and adds a residual-selected complement drawn from the Krylov subspace
of the two-site effective Hamiltonian. Each block is weighted by its propagator coefficient
and selected by a pivoted QR at ``rel_tol``, so a direction is added exactly when it would
survive the truncation that follows, and a bond is enlarged by no other direction.

Setting ``max_segment_bond`` or ``max_segment_width`` switches on segmented mode, which
bounds the transient peak bond by collapsing the chain periodically.

Example
-------
>>> config = RAGEMPSConfig(max_bond_dim=64, rel_tol=1e-10)
>>> solver = RAGE_MPS(state, hamiltonian, 0.01, 1.0, observables, config)
>>> solver.run()
"""

import numpy as np
from numpy.linalg import qr

from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.state_operator_contraction import (contract_leaf,
                                                               contract_any)
from .time_evolution import EvoDirection
from .bug_util import RAGEMPSConfig
from .effective_time_evolution import single_site_time_evolution
from .residual_common import ResidualSelectionMixin
from .BUG_MPS import BUG_MPS

__all__ = ["RAGE_MPS"]


class RAGE_MPS(ResidualSelectionMixin, BUG_MPS):
    """MPS integrator with residual-selected bond augmentation.

    Inherits the sweep, the chain-end Galerkin step and the environment cache from
    BUG_MPS; replaces the per-bond K-step with two-site residual selection and the
    post-sweep truncation with in-sweep collapses plus a final backward SVD. See the
    module docstring for the selection rule and RAGEMPSConfig for the knobs.

    Args:
        initial_state: The MPS to evolve. Copied by the base class.
        hamiltonian: The Hamiltonian as an MPO. Must be an AbstractTimeDepTTNO when
            ``config.time_dep`` is set.
        time_step_size: Step size ``dt``.
        final_time: Time at which the evolution stops.
        operators: Observables to measure during the run.
        config: A RAGEMPSConfig; defaults are used when omitted.

    Attributes:
        peak_bond: Largest bond dimension the sweep reached *before* truncation, as a
            running maximum over the whole run. This is the quantity segmentation bounds,
            and the only one that shows whether ``config.max_segment_bond`` did anything:
            the recorded ``max_bond_dim`` is the bond that survived the truncation, which
            is capped by ``config.max_bond_dim`` either way.
    """
    config_class = RAGEMPSConfig

    def __init__(self, initial_state, hamiltonian, time_step_size, final_time,
                 operators, config=None):
        # The engine comes from BUG_MPS: a sweeping walk cutting per bond. Segmentation
        # needs exactly that, since _truncate_window hands it one window of the chain
        # instead of the whole path.
        super().__init__(initial_state, hamiltonian, time_step_size,
                         final_time, operators, config)
        # Index in update_path where the last segment starts; 0 means the whole chain
        # is one segment. Set by the sweep, consumed by the post-sweep truncation.
        self._last_checkpoint = 0
        self._steps_done = 0        # completed sweeps; gates the warm-up
        self.peak_bond = 0          # widest pre-truncation bond, running max over the run
        self._warn_degenerate_aug_cap()

    # --------------------------------------------------- residual selection
    # The bond budget, the propagator column, the pivoted complement, the warm-up
    # count and the basis augmentation all live on ResidualSelectionMixin, shared
    # with RAGE_TTN. Only the chain-specific two-site action stays here.

    def _two_site_action(self, site_id: str, neighbor_id: str,
                         theta: np.ndarray) -> np.ndarray:
        """Apply the two-site effective Hamiltonian to the two-site tensor ``theta``.

        Args:
            site_id: Left site of the pair.
            neighbor_id: Right site of the pair.
            theta: Legs ``(D_left, d_site, D_right, d_neighbor)``; ``D_left`` is absent
                at the root and ``D_right`` at a leaf.

        Returns:
            The image of ``theta``, in the same layout, so the map can be iterated
            for the Krylov recurrence.
        """
        site_node = self.state.nodes[site_id]
        neigh_node = self.state.nodes[neighbor_id]
        w_site = self.hamiltonian.tensors[site_id]
        w_neigh = self.hamiltonian.tensors[neighbor_id]
        has_left = not site_node.is_root()
        neigh_children = neigh_node.children
        has_right = len(neigh_children) > 0
        if not has_left and not has_right:
            raise ValueError("Two-site residual needs at least three sites.")

        if has_left and has_right:
            # theta (Dl, d_site, Dr, d_neigh); w (w_left, w_right, out, in)
            left_env = self.partial_tree_cache.get_entry(site_node.parent, site_id)
            right_env = self.partial_tree_cache.get_entry(neigh_children[0], neighbor_id)
            x = np.tensordot(theta, left_env, axes=([0], [0]))
            x = np.tensordot(x, w_site, axes=([0, 3], [3, 0]))
            x = np.tensordot(x, w_neigh, axes=([1, 3], [3, 0]))
            x = np.tensordot(x, right_env, axes=([0, 3], [0, 1]))
            return np.transpose(x, (0, 1, 3, 2))         # (Dl', o_site, Dr', o_neigh)
        if has_left:
            # neighbour is a leaf: theta (Dl, d_site, d_neigh); w_neigh (w, out, in)
            left_env = self.partial_tree_cache.get_entry(site_node.parent, site_id)
            x = np.tensordot(theta, left_env, axes=([0], [0]))
            x = np.tensordot(x, w_site, axes=([0, 2], [3, 0]))
            return np.tensordot(x, w_neigh, axes=([0, 2], [2, 0]))   # (Dl', o_s, o_n)
        # site is the root: theta (d_site, Dr, d_neigh); w_site (w, out, in)
        right_env = self.partial_tree_cache.get_entry(neigh_children[0], neighbor_id)
        x = np.tensordot(theta, w_site, axes=([0], [2]))
        x = np.tensordot(x, w_neigh, axes=([1, 2], [3, 0]))
        x = np.tensordot(x, right_env, axes=([0, 2], [0, 1]))
        return np.transpose(x, (0, 2, 1))                # (o_site, Dr', o_neigh)

    def _augmented_bond_basis(self, site_id: str, right_neighbor_id: str,
                              matrix_form: np.ndarray, m_full: int,
                              first_site: bool) -> np.ndarray:
        """The augmented bond basis as an ``(m_full, k)`` matrix.

        The floor ``QR(old)`` extended by the two-site Krylov complement, selected at
        ``config.rel_tol`` and capped by ``config.max_aug_bond_dim``.

        Args:
            site_id: Site whose bond toward ``right_neighbor_id`` is being widened.
            right_neighbor_id: The right neighbour.
            matrix_form: ``old`` matricised as ``(m_full, r)``.
            m_full: Dimension of the complementary leg space, the ceiling on ``k``.
            first_site: True at the root, where the bond is leg 0 rather than leg 1.
        """
        floor, _ = qr(matrix_form)                       # (m_full, r)
        if self._bond_budget(floor.shape[1], m_full) <= floor.shape[1]:
            return floor                                 # degenerate: no complement
        old = self.state.tensors[site_id]
        neighbor_tensor = self.state.tensors[right_neighbor_id]
        bond_axis = 0 if first_site else 1
        theta = np.tensordot(old, neighbor_tensor, axes=([bond_axis], [0]))
        blocks = self._arnoldi_blocks(
            lambda t: self._two_site_action(site_id, right_neighbor_id, t),
            theta, m_full)
        return self._augment_basis(floor, blocks, m_full)

    # ------------------------------------------------------- basis updates
    def _update_first_site(self, site_id: str, right_neighbor_id: str) -> None:
        """Residual basis update at the root; tensor ``(D_right, d)``, no K-step."""
        self.state.move_orthogonalization_center(right_neighbor_id,
                                                 preserve_legs_order=True)
        old = self.state.tensors[site_id]                # (D_right, d)
        matrix_form = old.T                              # (d, D_right)
        m_full = matrix_form.shape[0]
        new_basis_tensor = self._augmented_bond_basis(
            site_id, right_neighbor_id, matrix_form, m_full, first_site=True).T
        self._install_basis(site_id, right_neighbor_id, old, new_basis_tensor)
        state_node, state_tensor = self.state[site_id]
        op_node, op_tensor = self.hamiltonian[site_id]
        env = contract_leaf(state_node, state_tensor, op_node, op_tensor)
        self.partial_tree_cache.add_entry(site_id, right_neighbor_id, env)

    def _update_middle_site(self, site_id: str, right_neighbor_id: str) -> None:
        """Residual basis update at an interior site; tensor ``(D_left, D_right, d)``."""
        self.state.move_orthogonalization_center(right_neighbor_id,
                                                 preserve_legs_order=True)
        old = self.state.tensors[site_id]                # (D_left, D_right, d)
        d_left, _, d_phys = old.shape
        matrix_form = np.transpose(old, (0, 2, 1)).reshape(d_left * d_phys, -1)
        m_full = d_left * d_phys
        new_basis_matrix = self._augmented_bond_basis(
            site_id, right_neighbor_id, matrix_form, m_full, first_site=False)
        k = new_basis_matrix.shape[1]
        new_basis_tensor = np.transpose(
            new_basis_matrix.reshape(d_left, d_phys, k), (0, 2, 1))
        self._install_basis(site_id, right_neighbor_id, old, new_basis_tensor)
        env = contract_any(node_id=site_id, next_node_id=right_neighbor_id,
                           state=self.state, operator=self.hamiltonian,
                           dictionary=self.partial_tree_cache)
        self.partial_tree_cache.add_entry(site_id, right_neighbor_id, env)

    def _augmentation_pass(self) -> None:
        """One left-to-right pass of basis updates, without the chain-end Galerkin step.

        The site updates only replace a tensor by a wider isometry and absorb the
        basis change into the right neighbour, so this pass widens bonds while leaving
        the represented state unchanged. Used for the warm-up passes.
        """
        path = self.update_path
        self._update_first_site(path[0], path[1])
        for i in range(1, len(path) - 1):
            self._update_middle_site(path[i], path[i + 1])

    # ------------------------------------------------------------ segments
    def _bond_dim(self, node_id: str, neigh_id: str) -> int:
        """The current dimension of the bond between ``node_id`` and ``neigh_id``."""
        node = self.state.nodes[node_id]
        return self.state.tensors[node_id].shape[node.neighbour_index(neigh_id)]

    def _scan_peak_bond(self) -> None:
        """Fold the widest bond of the whole chain into ``peak_bond``.

        The O(N) form, for the one-shot sweep and for the warm-up passes, where no bond
        is revisited and the state is at its widest once the pass ends. The segmented
        sweep cannot use it: a collapse shrinks the bonds behind the checkpoint before
        the sweep is over, so there the peak is taken bond by bond as each is augmented.
        """
        widest = 0
        for node_id in self.state.nodes:
            node = self.state.nodes[node_id]
            shape = self.state.tensors[node_id].shape
            for neigh_id in node.neighbouring_nodes():
                widest = max(widest, shape[node.neighbour_index(neigh_id)])
        self.peak_bond = max(self.peak_bond, widest)

    def _rebuild_segment_left_envs(self, path, a_index: int, b_index: int) -> None:
        """Recompute the left environments a collapse staled, ``a_index`` to ``b_index``.

        Right environments and earlier segments stay valid, so this costs O(segment)
        rather than a full cache rebuild.
        """
        for j in range(a_index, b_index):
            node_id, right_id = path[j], path[j + 1]
            if j == 0:
                state_node, state_tensor = self.state[node_id]
                op_node, op_tensor = self.hamiltonian[node_id]
                env = contract_leaf(state_node, state_tensor, op_node, op_tensor)
            else:
                env = contract_any(node_id=node_id, next_node_id=right_id,
                                   state=self.state, operator=self.hamiltonian,
                                   dictionary=self.partial_tree_cache)
            self.partial_tree_cache.add_entry(node_id, right_id, env)

    def _truncate_window(self, path, a_index: int, b_index: int) -> None:
        """Truncate bonds ``(a, a+1) .. (b-1, b)``.

        The engine sweeps ``path[b_index .. a_index]``, cutting each bond exactly once as
        the orthogonality centre crosses it, and leaves the centre at ``path[a_index]``.
        The mode is BUG_MPS's whole-chain one; only the path is the window, which
        unsegmented is the whole chain.

        Both the in-sweep collapse and the post-sweep finalisation route through here, so
        it is the sweep's only truncation point.

        On entry the state is left-canonical up to ``b_index`` with the centre there.
        """
        window = list(reversed(path[a_index:b_index + 1]))
        self.trunc_engine.truncate(self.state, self.config, path=window,
                                   preserve_legs_order=True)

    def _collapse_segment(self, path, a_index: int, b_index: int) -> None:
        """Collapse segment ``path[a_index .. b_index]``, centre at ``path[b_index]``.

        Evolve forward at the checkpoint to populate the segment, truncate back to its
        start, re-centre, then evolve backward. The backward solve is required: the SVD
        selects the evolved basis, and only ``exp(+i dt H)`` un-rotates it.

        This makes segmentation a distinct integrator
        rather than a re-bracketing of the one-shot sweep.
        """
        b_id = path[b_index]
        mode, options = self.config.local_solver()
        # (1) forward Galerkin at the checkpoint -- populate the segment.
        evolved = single_site_time_evolution(b_id, self.state, self.hamiltonian,
                                             self.time_step_size,
                                             self.partial_tree_cache, mode=mode,
                                             solver_options=options,
                                             forward=EvoDirection.FORWARD)
        self.state.replace_tensor(b_id, evolved, new_shape=True)
        # (2) truncate the reached basis, sweeping back to the segment start.
        self._truncate_window(path, a_index, b_index)
        # (3) carry the centre back to the checkpoint.
        self.state.move_orthogonalization_center(b_id, preserve_legs_order=True)
        # (4) refresh only the environments the collapse staled.
        self._rebuild_segment_left_envs(path, a_index, b_index)
        # (5) backward Galerkin -- un-rotate, leaving un-evolved amplitude.
        unevolved = single_site_time_evolution(b_id, self.state, self.hamiltonian,
                                               self.time_step_size,
                                               self.partial_tree_cache, mode=mode,
                                               solver_options=options,
                                               forward=EvoDirection.BACKWARD)
        self.state.replace_tensor(b_id, unevolved, new_shape=True)

    def _segment_triggers(self):
        """The active segmentation triggers as ``(max_bond, max_width)``.

        Either entry is ``None`` when that trigger is unset or disabled.
        """
        cap = self.config.max_segment_bond
        if cap is None or cap <= 0:
            cap = None
        width = self.config.max_segment_width
        if width is None or width <= 0 or width == float("inf"):
            width = None
        return cap, width

    # ----------------------------------------------------------- time step
    def sweeping_update(self) -> None:
        """Run one left-to-right sweep, segmented if a trigger is configured.

        Preceded by ``config.warmup_sweeps`` extra basis-update passes, each followed
        by a re-gauge to the chain start. Which steps get them is set by
        ``config.warmup_every_step``; see ResidualSelectionMixin._warmup_passes.
        """
        for _ in range(self._warmup_passes()):
            self._augmentation_pass()
            # Before the re-gauge: its QR would rewrite the bond dimensions being recorded.
            self._scan_peak_bond()
            self.prepare_next_sweep()
        cap, width = self._segment_triggers()
        if cap is None and width is None:
            super().sweeping_update()                    # one-shot RAGE
            self._scan_peak_bond()                       # widest bond, pre-truncation
            self._last_checkpoint = 0                    # whole chain is one segment
        else:
            self._segmented_sweep(cap, width)

    def _segmented_sweep(self, cap, width) -> None:
        """One sweep that collapses the chain whenever a segmentation trigger fires."""
        path = self.update_path
        n = len(path)
        a_index = 0                                      # current segment start
        for i, site_id in enumerate(path):
            if i == n - 1:
                # Chain-end Galerkin. No site follows it, so no checkpoint can open
                # here and the trigger is not consulted.
                self._update_last_site(site_id)
                break
            if i == 0:
                self._update_first_site(site_id, path[1])
            else:
                self._update_middle_site(site_id, path[i + 1])
            # Checkpoint past the segment start, so the collapse has a bond to compress.
            bond = self._bond_dim(site_id, path[i + 1])
            self.peak_bond = max(self.peak_bond, bond)
            checkpoint = i + 1
            if checkpoint < n - 1 and checkpoint > a_index and (
                    (cap is not None and bond > cap)
                    or (width is not None and checkpoint - a_index >= width)):
                self._collapse_segment(path, a_index, checkpoint)
                a_index = checkpoint
        self._last_checkpoint = a_index

    def _finalize_last_segment(self) -> None:
        """Truncate the last segment, re-gauge to site 0 and rebuild the cache.

        Earlier segments were already truncated by their collapses, so only the last
        one -- never collapsed before the chain-end Galerkin step -- is truncated here.
        With no interior checkpoint that is the whole chain.
        """
        path = self.update_path
        self._truncate_window(path, self._last_checkpoint, len(path) - 1)
        first_site = path[0]
        self.state.move_orthogonalization_center(first_site, preserve_legs_order=True)
        self.partial_tree_cache = SandwichCache.init_cache_but_one(
            self.state, self.hamiltonian, first_site)

    def run_one_time_step(self) -> None:
        """Advance the state by ``dt``.

        The sweep drives the truncation itself, through the in-sweep collapses and the
        finalisation below, rather than through a separate post-sweep pass. It is the same
        truncation, in the same mode and through the same engine, restricted to the segment
        each call compresses; unsegmented the two coincide.
        """
        self.sweeping_update()
        self._steps_done += 1                            # gates the warm-up
        self._finalize_last_segment()
