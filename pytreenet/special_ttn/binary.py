"""
Implements generation functions for binary Tree Tensor Network States (TTNS).

Two builders live here. :func:`generate_binary_ttns` produces a balanced binary tree over a
one-dimensional list of physical sites; :func:`optimised_2d_binary_ttn` produces a binary tree
over a square lattice, pairing along alternating directions so that lattice neighbours stay
close in the tree.
"""
import math
import re
from enum import Enum
from typing import Self

import numpy as np
from numpy import ndarray, zeros

from ..ttns import TreeTensorNetworkState
from ..core.node import Node
from ..util.ttn_exceptions import positivity_check
from .special_nodes import constant_bd_trivial_node

__all__ = ["generate_binary_ttns", "optimised_2d_binary_ttn"]


PHYS_PREFIX = "qubit"     # Prefix for physical nodes identifiers
VIRTUAL_PREFIX = "node"   # Prefix for virtual nodes identifiers

def _create_trivial_tensor_node(node_id: str,
                                bond_dim: int,
                                num_legs: int,
                                dtype=None) -> tuple[Node, ndarray]:
    """
    Args:
        node_id: Identifier for the new node
        bond_dim: Bond dimension for all legs
        num_legs: Number of legs for the tensor
        dtype: Data type for the tensor (defaults to the dtype of constant_bd_trivial_node)
        
    Returns:
        A tuple containing the created Node and its associated tensor
    """
    node = Node(identifier=node_id)
    tensor = constant_bd_trivial_node(bond_dim, num_legs)

    if dtype is not None:
        if (np.iscomplexobj(tensor)
                and not np.issubdtype(np.dtype(dtype), np.complexfloating)):
            # constant_bd_trivial_node always returns a COMPLEX array whose entries are
            # exactly 0 and 1, so the imaginary part is identically zero and dropping it is
            # lossless. Do it explicitly: astype would emit a ComplexWarning about discarding
            # an imaginary part that does not exist, on every real-dtype tree built.
            tensor = tensor.real
        tensor = tensor.astype(dtype)

    return node, tensor


def generate_binary_ttns(num_phys: int,
                         bond_dim: int,
                         phys_tensor: ndarray,
                         depth: int | None = None,
                         phys_prefix: str = "site",
                         dtype=np.complex128) -> TreeTensorNetworkState:
    """Generate a balanced binary tree tensor network state.

    Physical sites sit at the leaves and virtual nodes form a balanced binary tree above
    them. The physical sites are distributed evenly between the two subtrees at every
    branching, so a site count that is not a power of two yields a tree whose leaf depths
    differ by at most one.

    Args:
        num_phys: Number of physical sites.
        bond_dim: Bond dimension of the tree tensor network.
        phys_tensor: Tensor for physical sites.
        depth: Maximum depth of the binary tree. ``None`` uses ``ceil(log2(num_phys))``;
            ``0`` builds a chain with no virtual nodes.
        phys_prefix: Identifier prefix of the physical (leaf) nodes, which come out as
            ``f"{phys_prefix}{q}"``. The virtual nodes keep the ``"node{level}_{pos}"``
            scheme, which is what :func:`clean_inefficient_paths` discriminates on, so a
            ``phys_prefix`` beginning with ``VIRTUAL_PREFIX`` is rejected.
        dtype: Element type of every tensor. Defaults to ``complex128``, which is what the
            time-evolution stack upcasts to anyway. Pass ``float`` to keep the tree real,
            for a representation whose correctness depends on staying in real arithmetic.

    Returns:
        The generated TreeTensorNetworkState.

    Raises:
        ValueError: ``phys_prefix`` collides with the virtual-node prefix, or ``num_phys``
            or ``bond_dim`` is not positive.
    """
    positivity_check(num_phys, "number of physical sites")
    positivity_check(bond_dim, "bond dimension")
    if phys_prefix.startswith(VIRTUAL_PREFIX):
        raise ValueError(
            f"phys_prefix={phys_prefix!r} collides with the virtual-node prefix "
            f"{VIRTUAL_PREFIX!r}; clean_inefficient_paths tells physical from virtual nodes by "
            "that prefix, so the two must stay distinguishable.")
    phys_tensor = phys_tensor.astype(dtype)

    # Special case for single node
    if num_phys == 1:
        ttns = TreeTensorNetworkState()
        node_id = f"{PHYS_PREFIX}0"
        node = Node(identifier=node_id)
        ttns.add_root(node, phys_tensor.copy())
        return _rename_physical_nodes(ttns, num_phys, phys_prefix)

    # Special case: depth=0 -> generate a chain (MPS) with no virtual nodes
    if depth == 0:
        return _rename_physical_nodes(
            _generate_mps_chain(num_phys, phys_tensor, bond_dim), num_phys, phys_prefix)

    # Calculate required depth if not provided
    if depth is None:
        # For a binary tree, we need depth = ceil(log2(num_phys))
        depth = max(1, math.ceil(math.log2(num_phys)))

    # Create an empty TTNS
    ttns = TreeTensorNetworkState()

    # Initialize the root node
    root_id = f"{VIRTUAL_PREFIX}0_0"

    # Calculate number of physical sites a binary tree of this depth can hold
    max_phys_in_binary_tree = 2 ** depth

    # Determine if we need a chain structure at level 0 (for many physical sites)
    need_chain = num_phys > max_phys_in_binary_tree

    if need_chain:
        # Calculate number of level-0 chain nodes needed
        num_chain_nodes = math.ceil(num_phys / max_phys_in_binary_tree)

        # Root node connects to level 1 nodes and the next chain node
        root_node, root_tensor = _create_trivial_tensor_node(
            root_id, 1, 3,  # 2 for binary branch + 1 for chain
            dtype=phys_tensor.dtype
        )
        ttns.add_root(root_node, root_tensor)

        # Create chain nodes at level 0
        current_chain_id = root_id
        phys_per_branch = [min(max_phys_in_binary_tree, num_phys - i * max_phys_in_binary_tree)
                          for i in range(num_chain_nodes)]

        for i in range(1, num_chain_nodes):
            chain_id = f"{VIRTUAL_PREFIX}0_{i}"

            # Last chain node has 2 legs (prev + branch),
            # middle chain nodes have 3 legs (prev + next + branch)
            is_last = i == num_chain_nodes - 1
            num_legs = 2 if is_last else 3

            chain_node, chain_tensor = _create_trivial_tensor_node(
                chain_id, 1, num_legs,
                dtype=phys_tensor.dtype
            )

            # Connect to previous chain node - using compatible=False
            # to ignore dimension checks
            ttns.add_child_to_parent(
                chain_node,
                chain_tensor,
                0,  # Chain node's parent leg
                current_chain_id,
                2 if i == 1 else 1,  # Previous node's chain leg
                compatible=False
            )

            current_chain_id = chain_id
    else:
        # No chain needed, just use the root node for a simple binary tree
        num_legs = min(2, num_phys)  # At most 2 legs for children
        root_node, root_tensor = _create_trivial_tensor_node(
            root_id, 1, num_legs,
            dtype=phys_tensor.dtype
        )
        ttns.add_root(root_node, root_tensor)

        # Simple case - one branch from root
        phys_per_branch = [num_phys]

    # Now build balanced binary subtrees from each chain node or root
    current_phys_idx = 0
    chain_nodes = [f"{VIRTUAL_PREFIX}0_{i}" for i in range(len(phys_per_branch))]

    for i, chain_id in enumerate(chain_nodes):
        num_phys_this_branch = phys_per_branch[i]

        if num_phys_this_branch == 0:
            continue

        # Build a balanced binary subtree using bond_dim=1 for initial construction
        ttns = build_balanced_binary_subtree(
            ttns,
            chain_id,
            num_phys_this_branch,
            depth,
            1,  # Use bond_dim=1 for initial construction
            phys_tensor,
            current_phys_idx)

        current_phys_idx += num_phys_this_branch

    # Clean up inefficient nodes (particularly virtual nodes with only one child)
    ttns = clean_inefficient_paths(ttns)

    # Final step: pad all bonds to the desired bond dimension
    ttns.pad_bond_dimensions(bond_dim)

    return _rename_physical_nodes(ttns, num_phys, phys_prefix)


def _rename_physical_nodes(ttns: TreeTensorNetworkState,
                           num_phys: int,
                           phys_prefix: str) -> TreeTensorNetworkState:
    """Rename ``qubit{q}`` -> ``{phys_prefix}{q}`` in place, and return the tree.

    Applied after construction rather than threaded through the recursion: the builders parse
    their own virtual identifiers (``int(parent_id.split('_')[0].replace(VIRTUAL_PREFIX,
    ''))``) and :func:`clean_inefficient_paths` discriminates physical from virtual by prefix,
    so a prefix threaded through every helper would have to be threaded through that parsing
    too. Renaming at the boundary keeps the recursion on one fixed scheme and still gives
    callers the identifiers they asked for.
    """
    if phys_prefix == PHYS_PREFIX:
        return ttns
    for q in range(num_phys):
        ttns.change_node_identifier(f"{phys_prefix}{q}", f"{PHYS_PREFIX}{q}")
    return ttns

def _generate_mps_chain(num_phys: int,
                        phys_tensor: ndarray,
                        bond_dim: int) -> TreeTensorNetworkState:
    """Generate a simple Matrix Product State (MPS) chain with no virtual nodes."""
    ttns = TreeTensorNetworkState()

    # One-site case: just the physical tensor as a single node
    if num_phys == 1:
        single_id = f"{PHYS_PREFIX}0"
        single_node = Node(identifier=single_id)
        # Use phys_tensor directly if 1D or squeeze last dim if needed
        if phys_tensor.ndim == 1:
            single_tensor = phys_tensor.copy()
        else:
            single_tensor = phys_tensor.squeeze(0)
        ttns.add_root(single_node, single_tensor)
        return ttns

    # First boundary node: shape (1, phys_dim)
    phys_dim = phys_tensor.size if phys_tensor.ndim == 1 else phys_tensor.shape[-1]
    first_id = f"{PHYS_PREFIX}0"
    first_node = Node(identifier=first_id)
    first_tensor = np.zeros((1, phys_dim), dtype=phys_tensor.dtype)

    # Embed physical tensor at trivial virtual index
    if phys_tensor.ndim == 1:
        first_tensor[0, :] = phys_tensor
    else:
        first_tensor[0, :] = phys_tensor[0, :]
    ttns.add_root(first_node, first_tensor)

    # Middle nodes (1 to num_phys-2)
    for i in range(1, num_phys - 1):
        node_id = f"{PHYS_PREFIX}{i}"
        node = Node(identifier=node_id)
        mid_tensor = np.zeros((1, 1, phys_dim), dtype=phys_tensor.dtype)
        # Embed physical tensor at trivial virtual indices
        if phys_tensor.ndim == 1:
            mid_tensor[0, 0, :] = phys_tensor
        else:
            mid_tensor[0, 0, :] = phys_tensor[0, :]
        # Connect to previous
        prev_id = f"{PHYS_PREFIX}{i-1}"
        # prev is boundary for i==1, interior otherwise
        parent_leg = 0 if i == 1 else 1
        ttns.add_child_to_parent(node, mid_tensor, 0, prev_id, parent_leg)

    # Last boundary node: shape (1, phys_dim)
    last_id = f"{PHYS_PREFIX}{num_phys-1}"
    last_node = Node(identifier=last_id)
    last_tensor = np.zeros((1, phys_dim), dtype=phys_tensor.dtype)
    if phys_tensor.ndim == 1:
        last_tensor[0, :] = phys_tensor
    else:
        last_tensor[0, :] = phys_tensor[0, :]
    # Connect to previous node
    prev_id = f"{PHYS_PREFIX}{num_phys-2}"
    # prev is interior only if num_phys > 2
    parent_leg = 1 if num_phys > 2 else 0
    ttns.add_child_to_parent(last_node, last_tensor, 0, prev_id, parent_leg)

    # Pad all bond dimensions to the requested size
    ttns.pad_bond_dimensions(bond_dim)
    return ttns

def build_balanced_binary_subtree(ttns: TreeTensorNetworkState,
                               parent_id: str,
                               num_phys: int,
                               max_depth: int,
                               bond_dim: int,
                               phys_tensor: ndarray,
                               phys_start_idx: int) -> TreeTensorNetworkState:
    """Build a balanced binary subtree from a parent node.
    
    This recursive function builds a balanced binary tree structure by distributing
    physical nodes evenly across the tree branches.
    
    Args:
        ttns: The tree tensor network state to modify
        parent_id: ID of the parent node
        num_phys: Number of physical nodes to distribute in this subtree
        max_depth: Maximum depth allowed for the subtree
        bond_dim: Bond dimension (should be 1 for initial construction)
        phys_tensor: Tensor for physical nodes
        phys_start_idx: Starting index for physical node numbering
        
    Returns:
        Updated TTNS with subtree added
    """
    # Early exit if no physical nodes or parent doesn't exist
    if num_phys == 0 or parent_id not in ttns.nodes:
        return ttns

    # Extract parent information
    parent_node = ttns.nodes[parent_id]
    parent_open_legs = parent_node.open_legs

    if not parent_open_legs:
        # Parent has no open legs for children
        return ttns

    # Parse parent level and position
    parent_level = int(parent_id.split('_')[0].replace(VIRTUAL_PREFIX, ''))
    parent_pos = int(parent_id.split('_')[1])
    next_level = parent_level + 1

    # Case 1: Bottom of tree or few physical nodes - connect physical nodes directly
    if max_depth == 1 or num_phys <= 2:
        return _connect_physical_nodes_to_parent(
            ttns, parent_id, num_phys, bond_dim, phys_tensor, phys_start_idx, max_depth)

    # Case 2: For deeper trees, build a balanced binary structure
    # Distribute physical nodes between left and right subtrees
    num_children = min(2, len(parent_open_legs))

    if num_children == 1:
        phys_per_child = [num_phys]
    else:
        # Distribute evenly between the two subtrees
        left_phys = num_phys // 2
        right_phys = num_phys - left_phys
        phys_per_child = [left_phys, right_phys]

    # Create virtual child nodes
    current_phys_idx = phys_start_idx

    for i in range(num_children):
        if phys_per_child[i] == 0:
            continue

        # Calculate position for this child
        child_pos = 2 * parent_pos + i

        # Create virtual node ID
        child_id = f"{VIRTUAL_PREFIX}{next_level}_{child_pos}"

        # Calculate how many legs the child tensor needs
        if max_depth == 2:
            # Leaf virtual node - parent + physical children
            child_legs = 1 + min(2, phys_per_child[i])
        else:
            # Internal node - parent + virtual children
            child_legs = 1 + min(2, math.ceil(phys_per_child[i] / 2))

        # Create node with bond_dim for all dimensions
        child_node, child_tensor = _create_trivial_tensor_node(
            child_id, bond_dim, child_legs,
            dtype=phys_tensor.dtype
        )

        # Get available parent leg
        parent_leg = parent_open_legs[i if i < len(parent_open_legs) else -1]

        # Connect child to parent using compatible=False to ignore dimension checks
        ttns.add_child_to_parent(
            child_node,
            child_tensor,
            0,  # Child's parent leg
            parent_id,
            parent_leg,
            compatible=False
        )

        # Recursively build subtree from this child
        ttns = build_balanced_binary_subtree(
            ttns,
            child_id,
            phys_per_child[i],
            max_depth - 1,
            bond_dim,
            phys_tensor,
            current_phys_idx)

        current_phys_idx += phys_per_child[i]

    return ttns

def _connect_physical_nodes_to_parent(
    ttns: TreeTensorNetworkState,
    parent_id: str,
    num_phys: int,
    bond_dim: int,
    phys_tensor: ndarray,
    phys_start_idx: int,
    max_depth: int
) -> TreeTensorNetworkState:
    """Helper function to connect physical nodes directly to a parent node.
    
    If there are more physical nodes than available parent legs, creates an 
    intermediate node to handle the overflow.
    """
    parent_node = ttns.nodes[parent_id]
    parent_open_legs = parent_node.open_legs

    # Connect physical nodes directly to parent if possible
    phys_to_connect = min(len(parent_open_legs), num_phys)

    for i in range(phys_to_connect):
        phys_id = f"{PHYS_PREFIX}{phys_start_idx + i}"
        phys_node = Node(identifier=phys_id)

        # Get available leg from parent
        parent_leg = parent_open_legs[i]

        # Create physical tensor with appropriate dimensions
        if phys_tensor.ndim == 1:
            # For 1D tensors
            phys_dim = phys_tensor.size
            new_tensor = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
            new_tensor[0, :] = phys_tensor
        else:
            # For 2D+ tensors
            phys_dim = phys_tensor.shape[-1]
            new_tensor = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
            new_tensor[0, :] = phys_tensor[0, :]

        # Connect to parent using compatible=False
        ttns.add_child_to_parent(
            phys_node,
            new_tensor,
            0,  # Physical node's parent leg
            parent_id,
            parent_leg,
            compatible=False
        )

    # Return if we connected all physical nodes
    if phys_to_connect == num_phys:
        return ttns

    # Handle remaining physical nodes that couldn't be connected directly
    remaining_phys = num_phys - phys_to_connect
    start_idx = phys_start_idx + phys_to_connect

    # Parse parent level and position
    parent_level = int(parent_id.split('_')[0].replace(VIRTUAL_PREFIX, ''))
    parent_pos = int(parent_id.split('_')[1])
    next_level = parent_level + 1

    # Create an intermediate node to handle the remaining physical nodes
    intermediate_id = f"{VIRTUAL_PREFIX}{next_level}_{2 * parent_pos}"

    # Create tensor with enough legs for the remaining physical nodes
    intermediate_legs = 1 + min(2, remaining_phys)  # 1 for parent + up to 2 for children

    # Create intermediate node and connect to parent
    intermediate_node, intermediate_tensor = _create_trivial_tensor_node(
        intermediate_id, bond_dim, intermediate_legs,
        dtype=phys_tensor.dtype
    )

    # Connect to parent - use first available open leg
    ttns.add_child_to_parent(
        intermediate_node,
        intermediate_tensor,
        0,  # Intermediate's parent leg
        parent_id,
        parent_open_legs[0] if parent_open_legs else 0,  # Use first available open leg
        compatible=False
    )

    # Connect physical nodes to intermediate node
    for i in range(min(intermediate_legs - 1, remaining_phys)):
        phys_id = f"{PHYS_PREFIX}{start_idx + i}"
        phys_node = Node(identifier=phys_id)

        # Create physical tensor
        if phys_tensor.ndim == 1:
            phys_dim = phys_tensor.size
            phys_tensor_node = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
            phys_tensor_node[0, :] = phys_tensor
        else:
            phys_dim = phys_tensor.shape[-1]
            phys_tensor_node = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
            phys_tensor_node[0, :] = phys_tensor[0, :]

        # Connect to intermediate node
        ttns.add_child_to_parent(
            phys_node,
            phys_tensor_node,
            0,  # Physical node's parent leg
            intermediate_id,
            i + 1,  # Intermediate node's leg (skip parent leg)
            compatible=False
        )

    # Recursively handle any additional physical nodes if needed
    if remaining_phys > intermediate_legs - 1:
        return build_balanced_binary_subtree(
            ttns,
            intermediate_id,
            remaining_phys - (intermediate_legs - 1),
            max_depth - 1,
            bond_dim,
            phys_tensor,
            start_idx + (intermediate_legs - 1)
        )

    return ttns

def clean_inefficient_paths(ttns: TreeTensorNetworkState) -> TreeTensorNetworkState:
    """Clean up inefficient paths in the TTN structure.
    
    This function:
    1. Ensures all virtual nodes have at least one open leg
    2. Removes inefficient virtual nodes with only one child (bypassing them)
    
    Args:
        ttns: The Tree Tensor Network State to clean up
        
    Returns:
        The cleaned up TreeTensorNetworkState
    """
    # Pass 1: Ensure all virtual nodes have at least one open leg
    for node_id in list(ttns.nodes.keys()):
        if node_id.startswith(PHYS_PREFIX):
            continue  # Skip physical nodes

        node = ttns.nodes[node_id]

        # Check if node has no open legs
        if not node.open_legs and node_id in ttns.tensors:
            # Add an open leg to the tensor
            tensor = ttns.tensors[node_id]
            new_shape = list(tensor.shape) + [1]  # Add a dimension of size 1
            new_tensor = tensor.reshape(new_shape)
            ttns.tensors[node_id] = new_tensor

            # Update the node
            new_node = Node(identifier=node_id)
            new_node.link_tensor(new_tensor)
            new_node.parent = node.parent
            new_node.children = node.children.copy()
            ttns.nodes[node_id] = new_node

    # Pass 2: Identify inefficient nodes (virtual nodes with exactly one child and one parent)
    inefficient_nodes = []

    for node_id in list(ttns.nodes.keys()):
        if node_id.startswith(PHYS_PREFIX):
            continue  # Skip physical nodes

        node = ttns.nodes[node_id]

        # Check if node has exactly one child and is not the root
        if len(node.children) == 1 and node.parent is not None:
            child_id = node.children[0]
            inefficient_nodes.append((node_id, child_id))

    # Pass 3: Bypass inefficient nodes
    for inefficient_id, child_id in inefficient_nodes:
        # Skip if node has already been removed
        if inefficient_id not in ttns.nodes:
            continue

        parent_id = ttns.nodes[inefficient_id].parent
        if parent_id is None:
            continue  # Skip if node is root

        # Get the parent and child nodes
        parent_node = ttns.nodes[parent_id]
        child_node = ttns.nodes.get(child_id)

        if child_node is None:
            continue

        # Remove inefficient node from parent's children
        if inefficient_id in parent_node.children:
            parent_node.children.remove(inefficient_id)

        # Set child's parent to parent_id
        child_node.parent = parent_id

        # Add child to parent's children
        if child_id not in parent_node.children:
            parent_node.children.append(child_id)

        # Clean up the inefficient node
        inefficient_node = ttns.nodes[inefficient_id]
        inefficient_node.parent = None
        inefficient_node.children = []

        # Remove the inefficient node from the TTN
        if inefficient_id in ttns.tensors:
            del ttns.tensors[inefficient_id]
        if inefficient_id in ttns.nodes:
            del ttns.nodes[inefficient_id]

    return ttns


def create_non_root_virt_tensor(bond_dim: int) -> ndarray:
    """
    Creates a virtual tensor for a non-root node.
    """
    virt_tensor = constant_bd_trivial_node(bond_dim, 3)
    return virt_tensor

class Direction(Enum):
    X = 1
    Y = 2

    def opposite(self) -> Self:
        """
        Returns the opposite direction.
        """
        if self == Direction.X:
            return Direction.Y
        else:
            return Direction.X

    def virt_id_appendix(self,
                         along_index: int,
                         perp_index: int,
                         ) -> str:
        """
        Returns the appendix for a virtual node identifier in the given direction.
        """
        if self == Direction.X:
            return str(perp_index) + "_" + str(along_index)
        else:
            return str(along_index) + "_" + str(perp_index)

class BTTNLevel:
    """
    Helper class to store information about a level in the optimised 2D BTTN.
    """

    def __init__(self,
                 level: list[list[tuple[str,str] | tuple[str]]],
                 own_ids: list[list[str]],
                 pairing_direction: Direction = Direction.X
                 ) -> None:
        self.level = level
        self.own_ids = own_ids
        self.pairing_direction = pairing_direction
        self._y_size_value = len(level)
        if self._y_size_value > 0:
            self._x_size_value = len(level[0])
        else:
            self._x_size_value = 0

    def x_size(self) -> int:
        """
        Returns the size of the level in the x direction.
        """
        return self._x_size_value

    def y_size(self) -> int:
        """
        Returns the size of the level in the y direction.
        """
        return self._y_size_value

    def size_by_direction(self,
                          direction: Direction
                          ) -> int:
        """
        Returns the size of the level in the given direction.
        """
        if direction == Direction.X:
            return self.x_size()
        else:
            return self.y_size()

    def append_to_x_direction(self,
                           element: tuple[str,str] | tuple[str],
                           own_id: str,
                           new_x_row: bool = False
                           ) -> None:
        """
        Appends an element to the last x direction.

        Args:
            element (tuple[str,str] | tuple[str]): The element to append.
            own_id (str): The own identifier of the element.
            new_x_row (bool): Whether to start a new x row.

        Raises:
            ValueError: If the previous row is not complete when starting
                        a new x row, or if the current row is full when
                        appending to the current x row.
        """
        if new_x_row:
            if len(self.level) > 0 and len(self.level[-1]) != self.x_size():
                raise ValueError("Cannot start new x row, previous row not complete!")
            self.level.append([element])
            self.own_ids.append([own_id])
            self._y_size_value += 1
        else:
            if self.x_size() != 1 and len(self.level[-1]) >= self.x_size():
                errstr = "Cannot append to x direction, row full!\n"
                errstr += "Add new_x_row=True to start a new row."
                raise ValueError(errstr)
            self.level[-1].append(element)
            self.own_ids[-1].append(own_id)
        if self.y_size() == 1:
            self._x_size_value += 1

    def append_to_y_direction(self,
                              element: tuple[str,str] | tuple[str],
                              own_id: str,
                              new_y_column: bool = False
                              ) -> None:
        """
        Appends an element to the y direction at the given x index.

        Args:
            element (tuple[str,str] | tuple[str]): The element to append.
            own_id (str): The own identifier of the element.
            new_y_column (bool): Whether to start a new y column.
        """
        if new_y_column:
            if self.y_size() == 0:
                self.level.append([])
                self.own_ids.append([])
            if len(self.level[0]) > 0 and len(self.level) != self.y_size():
                raise ValueError("Cannot start new y column, previous column not complete!")
            self.level[0].append(element)
            self.own_ids[0].append(own_id)
            self._x_size_value += 1
        else:
            if self.x_size() == 1:
                self.level.append([element])
                self.own_ids.append([own_id])
            elif self.x_size() != 1 and len(self.level[-1]) == self.x_size():
                raise ValueError("Cannot append to y direction, column full!")
            else:
                for i, row in enumerate(self.level):
                    if len(row) < self.x_size():
                        row.append(element)
                        self.own_ids[i].append(own_id)
                        break
        if self.x_size() == 1:
            self._y_size_value += 1

    def append_element(self,
                       element: tuple[str,str] | tuple[str],
                       own_id: str,
                       direction: Direction,
                       new_line: bool = False
                       ) -> None:
        """
        Appends an element to the level in the given direction.

        Args:
            element (tuple[str,str] | tuple[str]): The element to append.
            own_id (str): The own identifier of the element.
            direction (Direction): The direction to append in.
            new_line (bool): Whether to start a new line in the given direction.
        """
        if direction == Direction.X:
            self.append_to_x_direction(element, own_id,
                                       new_x_row=new_line)
        else:
            self.append_to_y_direction(element, own_id,
                                       new_y_column=new_line)

    def get_nn_by_direction(self,
                            index_along_dir: int,
                            index_perp_dir: int,
                            direction: Direction
                            ) -> tuple[str,str] | tuple[str]:
        """
        Returns the nearest neighbours in the given direction at the given index.

        Args:
            index_along_dir (int): The index in the given direction.
            index_perp_dir (int): The index perpendicular to the given direction.
            direction (Direction): The direction to get the nearest neighbours in.

        Returns:
            list[tuple[str,str] | tuple[str]]: The nearest neighbours or a
                single site with the level's own identifiers.
        """
        is_last_in_dir = index_along_dir == self.size_by_direction(direction) - 1
        if direction == Direction.X:
            n1 = self.own_ids[index_perp_dir][index_along_dir]
            if not is_last_in_dir:
                n2 = self.own_ids[index_perp_dir][index_along_dir + 1]
                return (n1, n2)
        else:
            n1 = self.own_ids[index_along_dir][index_perp_dir]
            if not is_last_in_dir:
                n2 = self.own_ids[index_along_dir + 1][index_perp_dir]
                return (n1, n2)
        return (n1, )

    @classmethod
    def from_starting_params(cls,
                             lattice_size: int,
                             phys_prefix: str | list[list[str]],
                             virtual_prefix: str
                             ) -> Self:
        """
        Creates the bottom level of the BTTN from the lattice size.
        """
        # We combine the sites along the x direction first
        level: list[list[tuple[str] | tuple[str,str]]] = []
        own_ids: list[list[str]] = []
        for y in range(lattice_size):
            row: list[tuple[str] | tuple[str,str]] = []
            own_ids_row: list[str] = []
            for x in range(lattice_size):
                if isinstance(phys_prefix, list):
                    site_id = phys_prefix[y][x]
                else:
                    site_id = phys_prefix + str(y * lattice_size + x)
                if x % 2 == 0 and x != lattice_size - 1:
                    if isinstance(phys_prefix, list):
                        neigh_id = phys_prefix[y][x + 1]
                    else:
                        neigh_id = phys_prefix + str(y * lattice_size + (x + 1))
                    row.append((site_id, neigh_id))
                    own_ids_row.append(virtual_prefix + "lev1_" + str(y) + "_" + str(x // 2))
                elif x % 2 == 1:
                    continue
                else:
                    # In this case there is a last unpaired site
                    row.append((site_id,))
                    own_ids_row.append(site_id)
            level.append(row)
            own_ids.append(own_ids_row)
        return cls(level, own_ids=own_ids)

    @classmethod
    def from_previous_level(cls,
                            previous_level: Self,
                            virtual_prefix: str,
                            ) -> Self:
        """
        Creates a new level from the previous level.
        """
        pairing_direction = previous_level.pairing_direction.opposite()
        opposite_direction = pairing_direction.opposite()
        if previous_level.size_by_direction(pairing_direction) == 1:
            if previous_level.size_by_direction(opposite_direction) == 1:
                errstr = "Cannot create new level, previous level has size 1 in both directions!"
                raise ValueError(errstr)
            # This means there is nothing to do in the pairing direction
            # Thus we swap the directions
            pairing_direction, opposite_direction = opposite_direction, pairing_direction
        level = cls([], [], pairing_direction=pairing_direction)
        for i1 in range(previous_level.size_by_direction(opposite_direction)):
            for i2 in range(previous_level.size_by_direction(pairing_direction)):
                if i2 % 2 == 0:
                    nn = previous_level.get_nn_by_direction(i2, i1,
                                                            pairing_direction)
                    if len(nn) == 2:
                        new_id = virtual_prefix + pairing_direction.virt_id_appendix(i1, i2 // 2)
                    else:
                        new_id = nn[0]
                    level.append_element(nn, new_id,
                                         pairing_direction,
                                         new_line=(i2 == 0))
                else:
                    continue
        return level

    def to_dict(self) -> dict:
        """
        Converts the level to a dictionary.

        The keys are the own identifiers and the values are the
        tuples of nearest neighbour identifiers.
        """
        result: dict = {}
        for i, row in enumerate(self.level):
            for j, element in enumerate(row):
                own_id = self.own_ids[i][j]
                result[own_id] = element
        return result

def optimised_2d_binary_ttn(lattice_size: int,
                            initial_bond_dim: int,
                            phys_tensor: ndarray,
                            phys_prefix: str | list[list[str]] = "site",
                            virtual_prefix: str = "node"
                            ) -> TreeTensorNetworkState:
    """
    Generates an optimised binary tree tensor network state for a 2D lattice.

    Args:
        lattice_size (int): The size of the lattice (assumed square).
        initial_bond_dim (int): The bond dimension of the tree tensor network state.
        phys_tensor (ndarray): The tensor for the physical sites. Not included are
            the virtual bond dimension to the parent leg.
        phys_prefix (str | list[list[str]]): The prefix for the physical nodes.
            Alternatively, a 2D list of physical node identifiers can be given.
        virtual_prefix (str): The prefix for the virtual nodes.
    
    Returns:
        TreeTensorNetworkState: The generated tree tensor network state.
    """
    levels = []
    bottom_level = BTTNLevel.from_starting_params(lattice_size,
                                                  phys_prefix,
                                                  virtual_prefix)
    levels.append(bottom_level)
    current_level = bottom_level
    level_index = 2
    while (current_level.size_by_direction(Direction.X),
           current_level.size_by_direction(Direction.Y)) != (1, 1):
        level_virt_prefix = virtual_prefix + "lev" + str(level_index) + "_"
        current_level = BTTNLevel.from_previous_level(current_level,
                                                      level_virt_prefix)
        levels.append(current_level)
        level_index += 1
    # Now we can create the TTNS by running from the top level to the bottom
    phys_dim = phys_tensor.shape[0]
    ttns = TreeTensorNetworkState()
    current_level = levels[-1]
    root_id = current_level.own_ids[0][0]
    root_node = Node(identifier=root_id)
    root_tensor = constant_bd_trivial_node(initial_bond_dim, 2)
    ttns.add_root(root_node, root_tensor)
    for level in reversed(levels):
        dictionary = level.to_dict()
        for own_id, nn_ids in dictionary.items():
            own_node = ttns.nodes[own_id]
            for node_id in nn_ids:
                if not node_id in ttns.nodes:
                    node = Node(identifier=node_id)
                    if re.match(rf"^{phys_prefix}\d+$", node_id):
                        # This is a physical node
                        tensor = zeros((initial_bond_dim, phys_dim),
                                       dtype=phys_tensor.dtype)
                        tensor[0, :] = phys_tensor
                    else:
                        tensor = create_non_root_virt_tensor(initial_bond_dim)
                    ttns.add_child_to_parent(node, tensor, 0,
                                             own_id,
                                             own_node.lowest_open_leg())
    return ttns

