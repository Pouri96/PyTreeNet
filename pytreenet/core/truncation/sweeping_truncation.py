"""SVD truncation along a path, each bond cut once with the orthogonality centre on it.

For a whole tree, see :mod:`~pytreenet.core.truncation.node_truncation`.
"""
from numpy.linalg import svd

from pytreenet.core.canonical_form import split_svd_contract_sv_to_neighbour
from pytreenet.util.tensor_splitting import truncate_singular_values

from .truncation_util import move_orth_for_path


__all__ = ["sweeping_per_bond_truncation"]


def _bond_name(ttn, node_id: str, neighbour_id: str) -> str:
    """The bond's name: its lower node, as the recursive walk names it."""
    return neighbour_id if ttn.nodes[neighbour_id].parent == node_id else node_id


def _bond_spectrum(ttn, node_id: str, neighbour_id: str, params):
    """The singular values the cut keeps across the bond, with the centre on ``node_id``."""
    node = ttn.nodes[node_id]
    tensor = ttn.tensors[node_id]
    index = node.neighbour_index(neighbour_id)
    matrix = tensor.transpose([index] + [i for i in range(tensor.ndim) if i != index])
    values = svd(matrix.reshape(tensor.shape[index], -1), compute_uv=False)
    return values[:len(truncate_singular_values(values, params)[0])]


def sweeping_per_bond_truncation(ttn,
                            update_path,
                            orth_path,
                            svd_params,
                            preserve_legs_order: bool = False,
                            bond_params=None,
                            spectra=None):
    """Truncate the bonds along ``update_path``, leaving the centre at its last node.

    Args:
        ttn: The tree tensor network state to truncate, in place.
        update_path: The path to truncate. May be part of the network.
        orth_path: The orthogonalisation path along which to move the centre.
        svd_params: The parameters for the SVD truncation.
        preserve_legs_order: Restore each split node's neighbour order after the SVD.
        bond_params: Called as ``(ttn, node_id, neighbour_id)`` for each bond's parameters.
            Defaults to ``svd_params`` everywhere.
        spectra: Optional mapping, filled with each bond's kept singular values, keyed by
            the bond's lower node. Costs one extra SVD per bond.
    """
    if ttn.orthogonality_center_id is None:
        ttn.canonical_form(update_path[0], preserve_legs_order=preserve_legs_order)
    elif ttn.orthogonality_center_id != update_path[0]:
        ttn.move_orthogonalization_center(update_path[0],
                                          preserve_legs_order=preserve_legs_order)

    for update_index, node_id in enumerate(update_path):
        if update_index == 0:
            next_node_id = orth_path[update_index][0]
            params = (svd_params if bond_params is None
                      else bond_params(ttn, node_id, next_node_id))
            if spectra is not None:
                spectra[_bond_name(ttn, node_id, next_node_id)] = _bond_spectrum(
                    ttn, node_id, next_node_id, params)
            split_svd_contract_sv_to_neighbour(ttn, node_id, next_node_id, params,
                                               preserve_legs_order=preserve_legs_order)
            ttn.orthogonality_center_id = next_node_id
        elif update_index < len(orth_path):
            next_node_id = orth_path[update_index][0]
            current_orth_path = orth_path[update_index-1]
            move_orth_for_path(ttn, current_orth_path,
                               preserve_legs_order=preserve_legs_order)
            params = (svd_params if bond_params is None
                      else bond_params(ttn, node_id, next_node_id))
            if spectra is not None:
                spectra[_bond_name(ttn, node_id, next_node_id)] = _bond_spectrum(
                    ttn, node_id, next_node_id, params)
            split_svd_contract_sv_to_neighbour(ttn, node_id, next_node_id, params,
                                               preserve_legs_order=preserve_legs_order)
            ttn.orthogonality_center_id = next_node_id
    if ttn.orthogonality_center_id != update_path[-1]:
        raise RuntimeError(
            f"The sweep must leave the centre at {update_path[-1]!r}, "
            f"not {ttn.orthogonality_center_id!r}.")
