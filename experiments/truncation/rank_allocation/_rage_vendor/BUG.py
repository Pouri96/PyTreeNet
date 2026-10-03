"""BUG: the recursive rank-adaptive Basis-Update and Galerkin integrator for trees.

The original tree BUG of Ceruti, Lubich and Sulz (doi:10.1137/23M1545846), the published
reference for BUG_TTN and RAGE_TTN.

One time step is a post-order DFS: a node is updated after all its children, and the root
is evolved last. Siblings can be updated concurrently (``config.enable_parallel``).
``config.warmup_sweeps`` adds basis-only recursions before a step (see ``WarmupConfig``).

Example
-------
>>> config = BUGConfig(max_bond_dim=64, rel_tol=1e-10, enable_parallel=True)
>>> solver = BUG(tree_state, hamiltonian, 0.01, 1.0, observables, config)
>>> solver.run()
"""

import concurrent.futures
from copy import deepcopy
from typing import Dict, List, Union

from numpy import concat, tensordot

from pytreenet.contractions.sandwich_caching import SandwichCache, update_tree_cache
from pytreenet.contractions.state_operator_contraction import (contract_leaf,
                                                               contract_any)
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.core.leg_specification import LegSpecification
from pytreenet.core.truncation import TruncationEngine
from pytreenet.core.ttn import pull_tensor_from_different_ttn
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.ttno.time_dep_ttno import AbstractTimeDepTTNO

from .ttn_time_evolution import TTNTimeEvolution
from .bug_util import (basis_change_tensor_id,
                       reverse_basis_change_tensor_id,
                       compute_basis_change_tensor,
                       compute_new_basis_tensor,
                       warmup_passes,
                       BUGConfig)
from .common_bug import warn_nonhermitian_krylov, validate_config
from .effective_time_evolution import single_site_time_evolution
from .tensor_splitting import tensor_qr_decomposition

__all__ = ["BUG"]

Observables = Union[List[Union[TensorProduct, TreeTensorNetworkOperator]],
                    Dict[str, Union[TensorProduct, TreeTensorNetworkOperator]],
                    TensorProduct,
                    TreeTensorNetworkOperator]


class BUG(TTNTimeEvolution):
    """Recursive rank-adaptive BUG integrator for tree tensor network states.

    See the module docstring for the recursion and the parallel sibling updates.

    Args:
        initial_state: The tree state to evolve. Copied by the base class.
        hamiltonian: The Hamiltonian as a TTNO. Must be an
            AbstractTimeDepTTNO when ``config.time_dep`` is set.
        time_step_size: Step size ``dt``.
        final_time: Time at which the evolution stops.
        operators: Observables to measure during the run.
        config: A BUGConfig; defaults are used when omitted.

    Attributes:
        state: The current tree state.
        hamiltonian: The Hamiltonian TTNO.
        new_state: The tree under construction during a step.
        env_cache_old: Environment blocks of the pre-step tree.
        env_cache_new: Environment blocks of the tree under construction.
        basis_change_cache: Basis-change tensors of already-updated children,
            consumed when their parent is updated.
        trunc_engine: Performs the post-step truncation.
    """
    config_class = BUGConfig

    def __init__(self,
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TreeTensorNetworkOperator,
                 time_step_size: float,
                 final_time: float,
                 operators: Observables,
                 config: Union[BUGConfig, None] = None) -> None:
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
        self.state.ensure_root_orth_center()
        self.config: BUGConfig
        validate_config(self)
        warn_nonhermitian_krylov(self)

        self.basis_change_cache = PartialTreeCachDict()
        # An alias, not a copy; rebound to the freshly built state each step.
        self._old_basis_state = self.state
        # The recursion ends at the root, where the recursive truncation starts.
        self.trunc_engine = TruncationEngine.recursive()
        self._steps_done = 0            # completed steps; gates the warm-up

    # ------------------------------------------------------------ time step
    def recursive_update(self, galerkin: bool = True) -> None:
        """Rebuild the whole tree by one post-order DFS BUG recursion.

        Args:
            galerkin: Run the root Galerkin solve. ``False`` makes the recursion a
                basis-only pass: every bond widens and the represented state is unchanged.
        """
        self.env_cache_new = SandwichCache(state=None, hamiltonian=None)
        self.basis_change_cache = PartialTreeCachDict()
        self.root_update(self.state, galerkin=galerkin)
        self._old_basis_state = self.new_state
        self.state = self.new_state

    def run_one_time_step(self, **kwargs) -> None:
        """Advance the state by ``dt``: the warm-up, one recursion, then truncation."""
        for _ in range(warmup_passes(self.config, self._steps_done)):
            self.recursive_update(galerkin=False)
        self.recursive_update()
        self._steps_done += 1
        self.trunc_engine.truncate(self.state, self.config)

    # ------------------------------------------------------- node updates
    def update_leaf_node(self, node_id: str,
                         current_state: TreeTensorNetworkState) -> None:
        """Update a leaf: evolve it, widen its basis, split off the basis change.

        Writes the new basis and the basis-change tensor into ``new_state`` and the
        resulting parent block into ``env_cache_new``.

        Args:
            node_id: The leaf to update.
            current_state: A state whose orthogonality centre is this leaf.
        """
        mode, options = self.config.local_solver()
        updated_tensor = single_site_time_evolution(
            node_id=node_id, state=current_state, hamiltonian=self.hamiltonian,
            time_step_size=self.time_step_size, tensor_cache=self.env_cache_old,
            mode=mode, solver_options=options)
        old_basis_tensor = self._old_basis_state.tensors[node_id]

        # QR of the [old | evolved] leaf pair, transposed to (k, d_phys).
        concat_tensor = concat((old_basis_tensor, updated_tensor), axis=0)
        new_basis_tensor, _ = tensor_qr_decomposition(concat_tensor, (1,), (0,))
        new_basis_tensor = new_basis_tensor.T
        basis_change_tensor = tensordot(old_basis_tensor, new_basis_tensor.conj(),
                                        axes=([1], [1]))
        if not current_state.nodes[self.new_state.nodes[node_id].parent].is_root():
            self.basis_change_cache.add_entry(
                node_id, self._old_basis_state.nodes[node_id].parent,
                basis_change_tensor)
        parent_id = current_state.nodes[node_id].parent
        state_node_before, _ = self.new_state[node_id]
        self.new_state.split_node_replace(node_id=node_id,
                                          tensor_a=basis_change_tensor,
                                          tensor_b=new_basis_tensor,
                                          identifier_a=basis_change_tensor_id(node_id),
                                          identifier_b=node_id,
                                          legs_a=LegSpecification(parent_id, [], []),
                                          legs_b=LegSpecification(None, [], [1]),
                                          strict_checks=False)
        state_node, state_tensor = self.new_state[node_id]

        op_node, op_tensor = self.hamiltonian[node_id]

        child_block = contract_leaf(state_node, state_tensor, op_node, op_tensor)
        self.env_cache_new.add_entry(node_id, parent_id, child_block)

    def update_non_leaf_node(self, node_id: str,
                             current_state: TreeTensorNetworkState) -> None:
        """Update an interior node: recurse into the children, absorb them, then update.

        Once the children's basis-change tensors have been contracted in, the update
        is the same as update_leaf_node one level up.

        Args:
            node_id: The node to update.
            current_state: A state whose orthogonality centre is this node.
        """
        self._update_children(list(current_state.nodes[node_id].children), current_state)

        pull_tensor_from_different_ttn(old_ttn=current_state, new_ttn=self.new_state,
                                       node_id=node_id,
                                       mod_fct=reverse_basis_change_tensor_id)
        self.new_state.contract_all_children(node_id)
        parent_id = self.new_state.nodes[node_id].parent
        self.env_cache_new.add_entry(parent_id, node_id,
                                     self.env_cache_old.get_entry(parent_id, node_id))
        mode, options = self.config.local_solver()
        updated_tensor = single_site_time_evolution(
            node_id, self.new_state, self.hamiltonian, self.time_step_size,
            self.env_cache_new, mode=mode, solver_options=options)
        new_state_node = self.new_state.nodes[node_id]

        old_tensor = self.new_state.tensors[node_id]
        new_basis_tensor = compute_new_basis_tensor(node=new_state_node,
                                                    old_tensor=old_tensor,
                                                    updated_tensor=updated_tensor,
                                                    neighbour_id=new_state_node.parent)
        old_basis_node, old_basis_tensor = self._old_basis_state[node_id]

        basis_change_tensor = compute_basis_change_tensor(
            node_old=old_basis_node,
            node_new=new_state_node,
            tensor_old=old_basis_tensor,
            tensor_new=new_basis_tensor,
            basis_change_tensor_cache=self.basis_change_cache)

        if not current_state.nodes[old_basis_node.parent].is_root():
            self.basis_change_cache.add_entry(node_id, new_state_node.parent,
                                              basis_change_tensor)
        self.new_state.split_node_replace(
            node_id=node_id,
            tensor_a=basis_change_tensor,
            tensor_b=new_basis_tensor,
            identifier_a=basis_change_tensor_id(node_id),
            identifier_b=node_id,
            legs_a=LegSpecification(new_state_node.parent, [], []),
            legs_b=LegSpecification(None, new_state_node.children,
                                    new_state_node.open_legs),
            strict_checks=False)
        op_node, _ = self.hamiltonian[node_id]

        child_block = contract_any(
            node_id=node_id,
            next_node_id=basis_change_tensor_id(node_id),
            state=self.new_state,
            operator=self.hamiltonian,
            dictionary=self.env_cache_new,
            **self._basis_change_id_trafos(node_id, op_node))
        self.env_cache_new.add_entry(node_id, parent_id, child_block)

    def _basis_change_id_trafos(self, node_id: str, op_node) -> Dict[str, object]:
        """Identifier translations that let ``contract_any`` see through basis-change nodes.

        Mid-recursion the state interposes basis-change nodes on edges the operator
        does not have. These map identifiers in both directions, so either contraction
        order resolves its neighbours.
        """
        bc_suffix = basis_change_tensor_id("")
        bc_id = basis_change_tensor_id(node_id)
        state_neighbours = set(self.new_state.nodes[node_id].neighbouring_nodes())

        def id_trafo_op(ident: str) -> str:
            if ident == bc_id:
                return op_node.parent
            if ident.endswith(bc_suffix):
                return reverse_basis_change_tensor_id(ident)
            return ident

        def id_trafo_bra(ident: str) -> str:
            if ident in state_neighbours:
                return ident
            wrapped = basis_change_tensor_id(ident)
            return wrapped if wrapped in state_neighbours else ident

        return {"id_trafo_op": id_trafo_op, "id_trafo_bra": id_trafo_bra}

    def update_node(self, node_id: str,
                    parent_state: TreeTensorNetworkState,
                    target_state: TreeTensorNetworkState) -> None:
        """Update one node, dispatching on whether it is a leaf.

        Args:
            node_id: The node to update.
            parent_state: A state whose orthogonality centre is this node's parent.
            target_state: This node's own working copy, to be re-centred on it.
        """
        parent_id = parent_state.nodes[node_id].parent
        target_state.move_orthogonalization_center(node_id)
        # Module-level call: each parallel sibling holds its own cache, and this writes into
        # whichever object it is handed.
        update_tree_cache(self.env_cache_old, target_state, self.hamiltonian,
                          parent_id, node_id)
        if target_state.nodes[node_id].is_leaf():
            self.update_leaf_node(node_id=node_id, current_state=target_state)
        else:
            self.update_non_leaf_node(node_id=node_id, current_state=target_state)

    def _update_children(self, children: List[str],
                         current_state: TreeTensorNetworkState) -> None:
        """Update every child of a node, concurrently when ``config.enable_parallel``.

        Each child works on its own copy of the (child, parent) pair, so no locking is needed.

        Args:
            children: Identifiers of the sibling nodes to update.
            current_state: The state whose orthogonality centre is their parent.
        """
        def target_for(child_id: str) -> TreeTensorNetworkState:
            return current_state.deepcopy_parts(
                [child_id, current_state.nodes[child_id].parent])

        if self.config.enable_parallel and len(children) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(self.update_node, node_id=child_id,
                                           parent_state=current_state,
                                           target_state=target_for(child_id))
                           for child_id in children]
                concurrent.futures.wait(futures)
                for future in futures:
                    future.result()          # re-raise anything a worker swallowed
        else:
            for child_id in children:
                self.update_node(node_id=child_id, parent_state=current_state,
                                 target_state=target_for(child_id))

    def root_update(self, current_state: TreeTensorNetworkState,
                    galerkin: bool = True) -> None:
        """Update the root on ``new_state``: recurse into the children, then evolve.

        The root is the one node with no parent to hand a basis change to, so its
        evolved tensor is the updated root tensor -- the Galerkin step.

        Args:
            current_state: The current state, with the root as orthogonality centre.
            galerkin: Run that Galerkin step. ``False`` stops after the recursion, which
                leaves the represented state unchanged in a wider basis.
        """
        root_id = current_state.root_id
        if self.state.orthogonality_center_id:
            self.state.move_orthogonalization_center(root_id)
        else:
            self.state.canonical_form(root_id)
        self.new_state = deepcopy(current_state)
        self.env_cache_old = SandwichCache.init_cache_but_one(current_state,
                                                              self.hamiltonian, root_id)
        self._update_children(list(current_state.nodes[root_id].children), current_state)
        pull_tensor_from_different_ttn(old_ttn=current_state, new_ttn=self.new_state,
                                       node_id=root_id,
                                       mod_fct=reverse_basis_change_tensor_id)
        self.new_state.contract_all_children(root_id)
        if not galerkin:
            # Every node is now the isometry of its own [old | evolved] pair and the basis
            # change M = C_old Q* sits in its parent: M Q = C_old, the state is unchanged.
            return
        mode, options = self.config.local_solver()
        updated_tensor = single_site_time_evolution(
            root_id, self.new_state, self.hamiltonian, self.time_step_size,
            self.env_cache_new, mode=mode, solver_options=options)
        self.new_state.replace_tensor(root_id, updated_tensor)

    # ------------------------------------------------- time-dependent hooks
    def update_hamiltonian(self) -> None:
        """Advance a time-dependent Hamiltonian, apply the layer's state change, and restore
        root-canonical form. Called by ``run`` only when ``config.time_dep`` is set."""
        self.hamiltonian.update(self.time_step_size)
        self.hamiltonian.modify_state(self.state)
        if self.state.orthogonality_center_id is None:
            self.state.canonical_form(self.state.root_id)
        else:
            self.state.move_orthogonalization_center(self.state.root_id)
        self._old_basis_state = self.state
