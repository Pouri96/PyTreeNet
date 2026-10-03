"""Helpers that move the orthogonality centre along a truncation path."""

__all__ = ["move_orth_for_path", "find_orthogonalization_path"]


def move_orth_for_path(ttn, path, preserve_legs_order: bool = False):
    """Move the orthogonality centre along a path, one edge at a time. No cache is updated.

    Args:
        ttn: The tree tensor network to re-gauge, in place.
        path: Starts at the current centre, ends at the target. Empty is a no-op.
        preserve_legs_order: Restore each node's neighbour order after the move.

    Raises:
        ValueError: The centre is not at ``path[0]``.
    """
    if len(path) == 0:
        return
    if ttn.orthogonality_center_id != path[0]:
        raise ValueError(f"The orthogonality centre must be at {path[0]!r} to walk this "
                         f"path, not at {ttn.orthogonality_center_id!r}.")
    for node_id in path[1:]:
        ttn.move_orthogonalization_center(node_id,
                                          preserve_legs_order=preserve_legs_order)


def find_orthogonalization_path(state, trunc_path):
    """The orthogonalisation path for each consecutive pair of nodes in ``trunc_path``."""
    orth_path = []
    for i in range(len(trunc_path)-1):
        sub_path = state.path_from_to(trunc_path[i], trunc_path[i+1])
        orth_path.append(sub_path[1::])
    return orth_path
