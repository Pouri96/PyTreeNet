"""SVD truncation of a whole tree by a post-order walk ending at the root.

The orthogonality centre is moved onto each node, which then cuts its child legs one at a time
(sequentially truncated HOSVD), so every bond is cut once with the centre on its upper node.
"""
from copy import copy

from numpy import ndarray

from pytreenet.core.leg_specification import LegSpecification
from pytreenet.core.node import Node
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.util.tensor_splitting import truncated_tensor_svd, SVDParameters

__all__ = ["recursive_node_cut_truncation", "node_cut_children"]


def _identity_id(child_id: str, node_id: str) -> str:
    """The identifier of the identity inserted on the ``child_id``--``node_id`` bond."""
    return f"{child_id}_identity_{node_id}"


def _projector_id(node_id: str, child_id: str, star: bool) -> str:
    """The identifier of one half of the projector pair on that bond."""
    if star:
        return f"{node_id}_projectorstar_{child_id}"
    return f"{node_id}_projector_{child_id}"


def _child_projector(node: Node,
                     node_tensor: ndarray,
                     child_id: str,
                     svd_params: SVDParameters) -> tuple:
    """The kept left factor of the SVD across the ``child_id`` leg, legs ``(child, new)``.

    Returns:
        ``(projector, singular_values)``, the singular values kept.
    """
    other_legs = list(range(node.nlegs()))
    child_index = node.neighbour_index(child_id)
    other_legs.pop(child_index)
    projector, singular_values, _ = truncated_tensor_svd(node_tensor,
                                                         (child_index, ),
                                                         tuple(other_legs),
                                                         svd_params)
    return projector, singular_values


def _insert_child_projector(child_id: str,
                            node_id: str,
                            projector: ndarray,
                            tree: TreeTensorNetworkState) -> None:
    """Insert the projector and its conjugate on the ``node_id``--``child_id`` bond."""
    id_identity = _identity_id(child_id, node_id)
    tree.insert_identity(child_id, node_id, new_identifier=id_identity)
    tree.split_node_replace(id_identity,
                            projector.conj(),
                            projector.T,
                            _projector_id(node_id, child_id, True),
                            _projector_id(node_id, child_id, False),
                            LegSpecification(node_id, [], []),
                            LegSpecification(None, [child_id], []))


def _contract_one_child_projector(tree: TreeTensorNetworkState,
                                  node_id: str,
                                  child_id: str) -> None:
    """Absorb one projector pair: the conjugate half into the node, the other into the child,
    so the node stays the centre and the child an isometry."""
    tree.contract_nodes(node_id, _projector_id(node_id, child_id, True),
                        new_identifier=node_id)
    tree.contract_all_children(_projector_id(node_id, child_id, False),
                               new_identifier=child_id)


def node_cut_children(node_id: str,
                      tree: TreeTensorNetworkState,
                      svd_params: SVDParameters,
                      bond_params=None,
                      preserve_legs_order: bool = True,
                      spectra=None) -> None:
    """Cut every child leg of ``node_id``, one at a time. The parent leg is cut by the node above.

    Args:
        node_id: The node whose child bonds to cut.
        tree: The tree tensor network to truncate, in place.
        svd_params: The parameters to cut with, when ``bond_params`` is not given.
        bond_params: Called as ``(tree, node_id, child_id)`` for each bond's parameters.
        preserve_legs_order: Restore the neighbour order of every node the cut touches.
        spectra: Optional mapping, filled as ``spectra[child_id] = singular_values``.

    Raises:
        ValueError: The orthogonality centre is not at ``node_id``.
    """
    if tree.orthogonality_center_id != node_id:
        raise ValueError(f"The orthogonality centre must be at {node_id!r} to cut there, "
                         f"not at {tree.orthogonality_center_id!r}.")
    children = copy(tree.nodes[node_id].children)
    if not children:
        return                              # a leaf cuts nothing; its bond is cut above

    # Absorbing a projector reorders legs; record the order to restore it below.
    order_before = ({nid: copy(tree.nodes[nid].children)
                     for nid in [node_id, *children]} if preserve_legs_order else {})

    for child_id in children:
        # Re-read: the previous cut changed the tensor and its leg positions.
        node = tree.nodes[node_id]
        params = (svd_params if bond_params is None
                  else bond_params(tree, node_id, child_id))
        projector, singular_values = _child_projector(node, tree.tensors[node_id],
                                                      child_id, params)
        if spectra is not None:
            spectra[child_id] = singular_values
        _insert_child_projector(child_id, node_id, projector, tree)
        _contract_one_child_projector(tree, node_id, child_id)

    for nid, neighbour_order in order_before.items():
        if tree.nodes[nid].children != neighbour_order:
            tree.update_children_and_leg_permutation(nid, neighbour_order)


def recursive_node_cut_truncation(tree: TreeTensorNetworkState,
                                  svd_params: SVDParameters,
                                  bond_params=None,
                                  spectra=None) -> TreeTensorNetworkState:
    """Truncate the whole tree by a post-order walk of node cuts, ending at the root.

    Args:
        tree: The tree tensor network to truncate, in place.
        svd_params: The parameters to cut every bond with.
        bond_params: Called as ``(tree, node_id, child_id)`` for each bond's parameters.
            Defaults to ``svd_params`` everywhere.
        spectra: Optional mapping, filled with each bond's kept singular values, keyed by
            the bond's lower node.

    Returns:
        The truncated tree, with the centre at the root.
    """
    root_id = tree.root_id
    if tree.orthogonality_center_id is None:
        tree.canonical_form(root_id, preserve_legs_order=True)
    elif tree.orthogonality_center_id != root_id:
        # Moving the existing centre is one path; re-canonicalising would touch every node.
        tree.move_orthogonalization_center(root_id, preserve_legs_order=True)
    _node_cut_subtree(root_id, tree, svd_params, bond_params, spectra)
    tree.orthogonality_center_id = root_id
    return tree


def _node_cut_subtree(node_id: str,
                      tree: TreeTensorNetworkState,
                      svd_params: SVDParameters,
                      bond_params=None,
                      spectra=None) -> None:
    """Cut every bond below ``node_id``. The centre is at ``node_id`` before and after."""
    for child_id in copy(tree.nodes[node_id].children):
        if tree.nodes[child_id].is_leaf():
            continue
        tree.move_orthogonalization_center(child_id, preserve_legs_order=True)
        _node_cut_subtree(child_id, tree, svd_params, bond_params, spectra)
        tree.move_orthogonalization_center(node_id, preserve_legs_order=True)
    node_cut_children(node_id, tree, svd_params, bond_params=bond_params,
                      preserve_legs_order=True, spectra=spectra)
