"""The effective-Hamiltonian application, with the environment blocks split around it.

Applying an effective Hamiltonian means contracting a site tensor with its environment
blocks and the Hamiltonian tensor. Contracting a block into the ket replaces the virtual
leg of dimension ``chi`` toward that neighbour by the pair ``(m, chi)``, with ``m`` the
Hamiltonian bond dimension toward it. Doing so for every neighbour before the Hamiltonian
is applied builds an intermediate carrying the product of all Hamiltonian bond dimensions::

    peak = d * prod(m_i * chi_i)

Splitting the neighbours into one group contracted before the Hamiltonian tensor and one
after leaves only one of the two partial products standing at a time::

    peak = d * prod(chi_i) * max( prod_{i in S} m_i , prod_{i not in S} m_i )

which is minimised by balancing them.

Pending upstream as Drachier/PyTreeNet#33; once merged, import it from
``pytreenet.contractions.state_operator_contraction`` and delete this module.
"""

from functools import lru_cache

import numpy as np

from pytreenet.contractions.state_operator_contraction import (
    env_tensor_ham_leg_index, env_tensor_ket_leg_index)
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.core.node import Node

__all__ = ["contract_ket_ham_with_envs"]

#: Above this many neighbours the exact (exponential-time) split is replaced by a greedy one.
_MAX_EXACT_SPLIT_DEGREE = 12


def _node_operator_input_leg(node: Node) -> list[int]:
    """The legs of a Hamiltonian node corresponding to its input (ket) side."""
    n_neigh, n_open = node.nneighbours(), node.nopen_legs()
    return list(range(n_neigh + n_open // 2, n_neigh + n_open))


def contract_ket_ham_with_envs(ket_node: Node,
                               ket_tensor: np.ndarray,
                               ham_node: Node,
                               ham_tensor: np.ndarray,
                               dictionary: PartialTreeCachDict) -> np.ndarray:
    """Contract a state node and a Hamiltonian node with their environments, splitting the
    neighbours around the Hamiltonian tensor (:func:`_neighbours_before_ham`).

    Args:
        ket_node: The ket node.
        ket_tensor: The ket tensor.
        ham_node: The Hamiltonian node.
        ham_tensor: The Hamiltonian tensor.
        dictionary: The cache holding the already contracted subtrees.

    Returns:
        The contracted tensor, in the leg order of the ket node::

             _____                   _____
            |     |____        _____|     |
            |     |                 |     |
            |     |        |        |     |
            |     |     ___|__      |     |
            |     |    |      |     |     |
            |     |____|      |_____|     |
            |  C1 |    |   H  |     |  C2 |
            |     |    |______|     |     |
            |     |        |        |     |
            |     |     ___|__      |     |
            |     |    |      |     |     |
            |     |____|  A   |_____|     |
            |_____|    |______|     |_____|

    """
    neighbour_ids = ket_node.neighbouring_nodes()
    num_neighbours = len(neighbour_ids)
    nopen_ket = ket_node.nopen_legs()
    blocks = [dictionary.get_entry(neighbour_id, ket_node.identifier)
              for neighbour_id in neighbour_ids]
    ket_leg = env_tensor_ket_leg_index()
    ham_leg = env_tensor_ham_leg_index()
    before_ham = _neighbours_before_ham(tuple(block.shape[ham_leg]
                                              for block in blocks))
    after_ham = [index for index in range(num_neighbours)
                 if index not in before_ham]
    num_after = len(after_ham)
    # Every contraction removes one leg and appends the two legs left by the block. As
    # the neighbours to be contracted first are in ascending order, each of them lost one
    # leg index per block contracted before it.
    kethamblock = ket_tensor
    for position, index in enumerate(before_ham):
        kethamblock = np.tensordot(kethamblock, blocks[index],
                                   axes=([index - position], [ket_leg]))
    # The legs of the remaining neighbours are now in front, followed by the open legs and
    # by the (hamiltonian, bra) leg pairs left by the blocks. The Hamiltonian legs of
    # these pairs are contracted away here.
    block_legs = [num_after + nopen_ket + 2 * position
                  for position in range(len(before_ham))]
    block_legs.extend(range(num_after, num_after + nopen_ket))
    ham_legs = [ham_node.neighbour_index(neighbour_ids[index])
                for index in before_ham]
    ham_legs.extend(_node_operator_input_leg(ham_node))
    kethamblock = np.tensordot(kethamblock, ham_tensor,
                               axes=(block_legs, ham_legs))
    if num_after == 0:
        # No neighbour was left, so the legs are in the order of the ket node.
        return kethamblock
    # The Hamiltonian legs of the remaining neighbours follow the leg order of the
    # Hamiltonian node, which need not be the one of the ket node.
    after_in_ham_order = sorted(after_ham,
                                key=lambda index:
                                ham_node.neighbour_index(neighbour_ids[index]))
    ham_positions = {index: num_after + len(before_ham) + position
                     for position, index in enumerate(after_in_ham_order)}
    for index in after_ham:
        removed = ham_positions.pop(index)
        kethamblock = np.tensordot(kethamblock, blocks[index],
                                   axes=([0, removed], [ket_leg, ham_leg]))
        for other, position in ham_positions.items():
            ham_positions[other] = position - 1 - (position > removed)
    # Finally the legs are put into the order of the ket node, i.e. one leg per neighbour
    # followed by the open legs.
    nopen_ham = kethamblock.ndim - num_neighbours
    positions = {index: position for position, index in enumerate(before_ham)}
    positions.update({index: len(before_ham) + nopen_ham + position
                      for position, index in enumerate(after_ham)})
    permutation = [positions[index] for index in range(num_neighbours)]
    permutation.extend(range(len(before_ham), len(before_ham) + nopen_ham))
    return np.transpose(kethamblock, permutation)


@lru_cache(maxsize=128)
def _neighbours_before_ham(ham_dims: tuple[int, ...]) -> tuple[int, ...]:
    """The neighbours whose environment blocks go in before the Hamiltonian tensor.

    Balances the product of the Hamiltonian bond dimensions on each side, which minimises
    the peak intermediate. Contracting every block first is kept unless the peak drops.
    Cached, since the dimensions are constant during a run.

    Args:
        ham_dims: The Hamiltonian bond dimension of every neighbour's environment block,
            in the leg order of the ket node.

    Returns:
        The indices of the neighbours to contract before the Hamiltonian tensor, ascending.
    """
    num_neighbours = len(ham_dims)
    total = 1
    for dim in ham_dims:
        total *= dim
    if num_neighbours > _MAX_EXACT_SPLIT_DEGREE:
        # Greedy fall-back: take the largest dimensions until their product reaches the
        # square root of the total.
        before, chosen = 1, []
        for index in sorted(range(num_neighbours),
                            key=lambda index: ham_dims[index],
                            reverse=True):
            if before * before <= total:
                before *= ham_dims[index]
                chosen.append(index)
        return tuple(sorted(chosen))
    best_indices, best_peak = tuple(range(num_neighbours)), total
    for subset in range(1 << num_neighbours):
        before = 1
        for index, dim in enumerate(ham_dims):
            if subset >> index & 1:
                before *= dim
        peak = max(before, total // before)
        if peak < best_peak:
            best_peak = peak
            best_indices = tuple(index for index in range(num_neighbours)
                                 if subset >> index & 1)
    return best_indices
