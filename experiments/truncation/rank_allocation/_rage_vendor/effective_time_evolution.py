"""Local time evolution under effective Hamiltonians (one site, one bond or two sites).

Where the solver allows it, the effective Hamiltonian is applied as a contraction and never
built densely. The single-site functions are defined here so that contraction uses
:func:`rage.util.state_operator_contraction.contract_ket_ham_with_envs`; the bond and two-site
functions are PyTreeNet's.
"""
from typing import Any

from numpy import ndarray
from numpy.typing import NDArray
import numpy as np

from pytreenet.contractions.effective_hamiltonians import (
    get_effective_single_site_hamiltonian_nodes)
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.core.node import Node
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.time_evolution.time_evolution import EvoDirection, TimeEvoMode
from pytreenet.time_evolution.time_evo_util.effective_time_evolution import (
    bond_time_evolution,
    two_site_time_evolution,
    effective_bond_evolution,
    effective_two_site_evolution,
)

from .state_operator_contraction import contract_ket_ham_with_envs

__all__ = ["bond_time_evolution", "single_site_time_evolution", "two_site_time_evolution",
           "effective_bond_evolution", "effective_single_site_evolution",
           "effective_two_site_evolution"]


def effective_single_site_evolution(
        state_tensor: NDArray[np.complex128],
        state_node: Node,
        hamiltonian_tensor: NDArray[np.complex128],
        hamiltonian_node: Node,
        time_step_size: float,
        tensor_cache: PartialTreeCachDict,
        mode: TimeEvoMode = TimeEvoMode.FASTEST,
        forward: EvoDirection = EvoDirection.FORWARD,
        **options
    ) -> NDArray[np.complex128]:
    """Evolve one site under its effective Hamiltonian.

    Args:
        state_tensor: The tensor of the state to be updated.
        state_node: The node of the state tensor.
        hamiltonian_tensor: The tensor of the Hamiltonian.
        hamiltonian_node: The node of the Hamiltonian tensor.
        time_step_size: The time step size.
        tensor_cache: The cache holding the neighbour blocks.
        mode: The mode of the time evolution. Where the mode allows it the effective
            Hamiltonian is applied as a contraction and never built densely.
        forward: Whether to evolve forward or backward.
        **options: Additional options for the solver; see ``TimeEvoMode``.

    Returns:
        The updated state tensor.
    """
    if mode.action_evolvable():
        def contraction_action(t, y):
            """The effective-Hamiltonian application driving the solver."""
            return contract_ket_ham_with_envs(state_node,
                                              y,
                                              hamiltonian_node,
                                              hamiltonian_tensor,
                                              tensor_cache)
        return mode.time_evolve_action(state_tensor,
                                       contraction_action,
                                       time_step_size,
                                       forward=forward,
                                       **options)
    # Fall back to the dense effective Hamiltonian.
    ham_eff = get_effective_single_site_hamiltonian_nodes(state_node,
                                                          hamiltonian_node,
                                                          hamiltonian_tensor,
                                                          tensor_cache)
    return mode.time_evolve(state_tensor,
                            ham_eff,
                            time_difference=time_step_size,
                            forward=forward,
                            **options)


def single_site_time_evolution(node_id: str,
                               state: TreeTensorNetworkState,
                               hamiltonian: TreeTensorNetworkOperator,
                               time_step_size: float,
                               tensor_cache: PartialTreeCachDict,
                               forward: EvoDirection = EvoDirection.FORWARD,
                               mode: TimeEvoMode = TimeEvoMode.FASTEST,
                               solver_options: dict[str, Any] | None = None
                               ) -> ndarray:
    """Evolve the site ``node_id`` of ``state`` under its effective Hamiltonian.

    Args:
        node_id: The identifier of the site to evolve.
        state: The state of the system.
        hamiltonian: The Hamiltonian of the system as a TTNO.
        time_step_size: The time step size.
        tensor_cache: The cache holding the neighbour blocks.
        forward: Whether to evolve forward or backward.
        mode: The mode of the time evolution.
        solver_options: Additional options for the solver; see ``TimeEvoMode``.

    Returns:
        The updated tensor of the site. The state is not written to.
    """
    if solver_options is None:
        solver_options = {}
    state_node, state_tensor = state[node_id]
    ham_node, ham_tensor = hamiltonian[node_id]
    return effective_single_site_evolution(
        state_tensor,
        state_node,
        ham_tensor,
        ham_node,
        time_step_size,
        tensor_cache,
        mode=mode,
        forward=forward,
        **solver_options
    )
