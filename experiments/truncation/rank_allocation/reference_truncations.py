"""The two reference truncations of Table I, kept here only to regenerate the table.

Both cut root-to-leaves with the orthogonality centre fixed at the root:

    carry_environments=False   Ref. [Ceruti2024, Alg. 7]
    carry_environments=True    Ref. [Grasedyck2010, Sec. 4]: the square root of each parent
                               bond's environment is contracted in before the cut, so the
                               singular values are the state's Schmidt values.
"""
from __future__ import annotations

from copy import copy
from typing import Union

from numpy import clip, moveaxis, ndarray, sqrt, tensordot
from numpy.linalg import eigh

from pytreenet.core.leg_specification import LegSpecification
from pytreenet.core.node import Node
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.util.tensor_splitting import SVDParameters, truncated_tensor_svd

__all__ = ["recursive_truncation"]


def identity_id(child_id: str, node_id: str) -> str:
    """The identifier for the identity tensor."""
    return f"{child_id}_identity_{node_id}"


def projector_identifier(child_id: str, node_id: str, star: bool) -> str:
    """The identifier for the projector tensor."""
    if star:
        return f"{child_id}_projectorstar_{node_id}"
    return f"{child_id}_projector_{node_id}"


def get_truncation_projector(node: Node,
                             node_tensor: ndarray,
                             child_id: str,
                             svd_parameters: SVDParameters) -> ndarray:
    """The projector that truncates ``node`` for the leg to ``child_id``.

    Leg order ``(child_leg, new_leg)``.
    """
    other_legs = list(range(node.nlegs()))
    child_index = node.neighbour_index(child_id)
    other_legs.pop(child_index)
    projector, _, _ = truncated_tensor_svd(node_tensor,
                                           (child_index, ),
                                           tuple(other_legs),
                                           svd_parameters)
    return projector


def insert_projection_operator_and_conjugate(child_id: str,
                                             node_id: str,
                                             projector: ndarray,
                                             tree: TreeTensorNetwork) -> None:
    """Insert the projector and its conjugate between two nodes."""
    id_identity = identity_id(child_id, node_id)
    tree.insert_identity(child_id, node_id, new_identifier=id_identity)
    proj_star_legs = LegSpecification(node_id, [], [])
    proj_legs = LegSpecification(None, [child_id], [])
    tree.split_node_replace(id_identity,
                            projector.conj(),
                            projector.T,
                            projector_identifier(node_id, child_id, True),
                            projector_identifier(node_id, child_id, False),
                            proj_star_legs,
                            proj_legs)


def environment_square_root(environment: ndarray) -> ndarray:
    """The Hermitian square root of a positive semi-definite environment."""
    eigenvalues, eigenvectors = eigh(environment)
    eigenvalues = clip(eigenvalues.real, 0, None)
    return (eigenvectors * sqrt(eigenvalues)) @ eigenvectors.conj().T


def weight_node_tensor(node: Node,
                       node_tensor: ndarray,
                       environment: Union[ndarray, None]) -> ndarray:
    """Contract the square root of the environment into the parent leg.

    The singular values of the result are those of the matricisation of the REPRESENTED
    tensor at that bond. The weighted tensor is used only to find the projectors and never
    enters the tree. ``environment is None`` -- the root, and every node when the
    environments are not carried -- returns the tensor unchanged.
    """
    if environment is None:
        return node_tensor
    parent_index = node.neighbour_index(node.parent)
    weight = environment_square_root(environment.conj())
    weighted_tensor = tensordot(weight, node_tensor, axes=(1, parent_index))
    return moveaxis(weighted_tensor, 0, parent_index)


def get_child_environment(node: Node,
                          node_tensor: ndarray,
                          environment: Union[ndarray, None],
                          child_id: str) -> ndarray:
    """The environment of the bond to ``child_id``.

    Everything on the root side of the bond contracted with its conjugate, found from the
    parent bond's environment and this node's tensor, so one contraction per node carries the
    environments from the root to the leaves.
    """
    weighted_tensor = weight_node_tensor(node, node_tensor, environment)
    child_index = node.neighbour_index(child_id)
    contracted_legs = tuple(leg for leg in range(weighted_tensor.ndim)
                            if leg != child_index)
    return tensordot(weighted_tensor, weighted_tensor.conj(),
                     axes=(contracted_legs, contracted_legs))


def truncate_node(node_id: str,
                  tree: TreeTensorNetwork,
                  svd_params: SVDParameters,
                  environment: Union[ndarray, None] = None,
                  carry_environments: bool = True) -> None:
    """Cut every child leg of ``node_id``, then recurse into the children."""
    node = tree.nodes[node_id]
    orig_children = copy(node.children)
    weighted_tensor = weight_node_tensor(node, tree.tensors[node_id], environment)
    for child_id in orig_children:
        projector = get_truncation_projector(node, weighted_tensor, child_id, svd_params)
        insert_projection_operator_and_conjugate(child_id, node_id, projector, tree)
    tree.contract_all_children(node_id)
    # Contract the projectors into the former children.
    for child_id in copy(tree.nodes[node_id].children):
        child_node = tree.nodes[child_id]
        assert len(child_node.children) == 1, "Projector node has more than one child!"
        orig_child_id = child_node.children[0]
        tree.contract_all_children(child_id, new_identifier=orig_child_id)
    node = tree.nodes[node_id]
    for child_id in orig_children:
        child_environment = (get_child_environment(node, tree.tensors[node_id],
                                                   environment, child_id)
                             if carry_environments else None)
        truncate_node(child_id, tree, svd_params, child_environment, carry_environments)


def recursive_truncation(tree: TreeTensorNetwork,
                         svd_params: SVDParameters,
                         carry_environments: bool = True) -> TreeTensorNetwork:
    """Truncate from the root to the leaves, the orthogonality centre pinned at the root.

    Args:
        tree: The state, truncated in place.
        svd_params: The SVD parameters; ``max_bond_dim`` caps every bond.
        carry_environments: ``True`` reproduces arm (b), the recursion carrying the
            environments; ``False`` reproduces arm (a), the recursion without them.

    Returns:
        The same tree.
    """
    root_id = tree.root_id
    if root_id != tree.orthogonality_center_id or tree.orthogonality_center_id is None:
        tree.canonical_form(root_id)
    truncate_node(root_id, tree, svd_params, None, carry_environments)
    return tree
