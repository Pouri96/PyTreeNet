"""Configuration classes and basis-update helpers for the BUG and RAGE integrators.

Each integrator has its own config, so a knob it does not read is a TypeError:

    IntegratorConfig(TTNTimeEvolutionConfig, SVDParameters)
    BUGConfig(WarmupConfig, ...)                                         + parallelism
    BUGMPSConfig(SweepCompositionConfig, ...)
    BUGTTNConfig(SweepCompositionConfig, ...)
    RAGEMPSConfig(ResidualSelectionConfig, ...)                          + segmentation
    RAGETTNConfig(ResidualSelectionConfig, ...)                          + max_node_size

where ``...`` is ``ParameterBudgetConfig, IntegratorConfig``. Every config is also an
``SVDParameters``, passed directly as ``svd_params`` to the truncation.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from numpy import ndarray, concatenate, transpose

from pytreenet.contractions.contraction_util import get_equivalent_legs
from pytreenet.contractions.state_state_contraction import contract_any_nodes
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.core.canonical_form import _build_leg_specs
from pytreenet.core.node import Node


from pytreenet.time_evolution.time_evolution import TimeEvoMode
from .ttn_time_evolution import TTNTimeEvolutionConfig
from .tensor_splitting import tensor_qr_decomposition, SVDParameters


__all__ = [
    "IntegratorConfig", "ResidualSelectionConfig", "WarmupConfig", "warmup_passes",
    "ParameterBudgetConfig",
    "BUGConfig", "BUGMPSConfig", "BUGTTNConfig", "RAGEMPSConfig",
    "RAGETTNConfig", "basis_change_tensor_id",
    "reverse_basis_change_tensor_id", "compute_new_basis_tensor",
    "compute_basis_change_tensor", "adjust_node1_structure_to_node2",
    "adjust_ttn1_structure_to_ttn2"
]


def _validate_residual_knobs(config) -> None:
    """Validate the residual-selection knobs shared through ``ResidualSelectionConfig``.

    Raises:
        ValueError: ``max_aug_bond_dim`` was given as a string.
    """
    # Read as a number deep in the sweep, where a string surfaces far from its cause.
    if isinstance(config.max_aug_bond_dim, str):
        raise ValueError(
            "max_aug_bond_dim must be a number or None. Got "
            f"{config.max_aug_bond_dim!r}; use float('inf') for no cap.")


def _validate_parameter_budget(config) -> None:
    """Validate the parameter budget against the rule that has to honour it.

    Raises:
        ValueError: The budget is not a positive integer, or ``random=True``.
    """
    budget = config.parameter_budget
    if budget is None:
        return
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(
            f"parameter_budget must be a positive int or None, got {budget!r}.")
    if config.random:
        raise ValueError(
            "parameter_budget cannot be used with random=True: the allocation is solved "
            "from the singular values the measuring walk reports, and a randomized SVD "
            "does not report the state's own, so the ranks would be decided from the "
            "wrong spectra.")


def _validate_warmup_knobs(config) -> None:
    """Coerce ``warmup_sweeps`` to a non-negative int.

    Raises:
        ValueError: ``warmup_sweeps`` is negative.
    """
    if int(config.warmup_sweeps) < 0:
        raise ValueError(f"warmup_sweeps must be >= 0, got {config.warmup_sweeps}!")
    config.warmup_sweeps = int(config.warmup_sweeps)


def _validate_segmentation_knobs(config) -> None:
    """Validate the chain segmentation triggers, which only RAGEMPSConfig carries.

    Raises:
        ValueError: ``max_segment_bond`` was given as a string.
    """
    if isinstance(config.max_segment_bond, str):
        raise ValueError(
            "max_segment_bond must be a number or None, got "
            f"{config.max_segment_bond!r}. Segmentation is opt-in: pass a bond dimension "
            "to enable it, or leave it None to sweep one-shot.")


def _validate_node_size(config) -> None:
    """Validate the adaptive-topology budget, which only RAGETTNConfig carries.

    Raises:
        ValueError: ``max_node_size`` was given as a string.
    """
    if isinstance(config.max_node_size, str):
        raise ValueError(
            "max_node_size must be a number or None. "
            f"Got {config.max_node_size!r}.")


@dataclass
class IntegratorConfig(TTNTimeEvolutionConfig, SVDParameters):
    """The knobs every integrator reads: truncation (SVDParameters), diagnostics
    (TTNTimeEvolutionConfig) and ``hermitian``. The local solver is set per integrator.

    Attributes:
        hermitian: Whether ``H_eff`` is Hermitian. True (the default) covers real- and
            imaginary-time evolution of a Hermitian ``H``. False with KRYLOV warns.
    """
    hermitian: bool = True

    def _solver_pair(self, mode: TimeEvoMode) -> tuple[TimeEvoMode, Dict[str, Any]]:
        """The ``(mode, options)`` pair for a local solver, with ``hermitian`` merged in.

        Returns a fresh dict each call, because the solver pops keys from it.
        """
        options = dict(self.solver_options)
        if 'hermitian' in TimeEvoMode.valid_solver_options(mode):
            options.setdefault('hermitian', self.hermitian)
        return mode, options

    def _resolve_solver_options(self, mode: TimeEvoMode, field_name: str) -> None:
        """Fill the mode's default options, then overlay the caller's.

        Raises:
            ValueError: ``solver_options`` holds a key the chosen mode does not read.
        """
        valid = TimeEvoMode.valid_solver_options(mode)
        unknown = set(self.solver_options) - valid
        if unknown:
            raise ValueError(
                f"solver_options {sorted(unknown)} are not read by "
                f"{field_name}={mode}. It reads {sorted(valid)}. A key it "
                "does not read would be dropped in the solver, leaving the run at a "
                "tolerance that was never applied.")
        # A partial dict must not leave other keys at the solver's looser defaults.
        resolved = TimeEvoMode.default_solver_options(mode)
        resolved.update(self.solver_options)
        self.solver_options = resolved


class SweepComposition(Enum):
    """How one time step is built out of basis-update sweeps.

    Attributes:
        SINGLE: One sweep of the full step from one endpoint.
        DOUBLE_ENDPOINT: Two half-sweeps of ``dt / 2`` from opposite endpoints, each
            followed by compression. Symmetric on a chain; on a branching tree the two
            sweeps are not mirror images.
    """

    SINGLE = "single"
    DOUBLE_ENDPOINT = "double_endpoint"


@dataclass
class SweepCompositionConfig:
    """The sweep-composition knob, read by the sweeping integrators BUG_MPS and BUG_TTN.

    Attributes:
        composition: How one step is assembled from sweeps. Defaults to
            ``SweepComposition.DOUBLE_ENDPOINT``; pass ``SweepComposition.SINGLE`` for the
            one-sweep step.
    """

    composition: SweepComposition = SweepComposition.DOUBLE_ENDPOINT


@dataclass
class ParameterBudgetConfig:
    """A size target in parameters, read by every integrator.

    Where ``max_bond_dim`` caps each rank, this caps the total size: each truncation solves the
    exact rank allocation for the target (:mod:`~pytreenet.core.truncation.parameter_budget`),
    at the cost of two truncation walks. If both are set, whichever cuts more applies.

    Attributes:
        parameter_budget: Entries the network may hold after a truncation, physical legs
            included. ``None`` (the default) disables it. Not allowed with ``random=True``.
    """
    parameter_budget: Optional[int] = None


@dataclass
class WarmupConfig:
    """The warm-up knobs, read by BUG, BUG_MPS, RAGE_MPS and RAGE_TTN.

    A warm-up pass is a basis-update pass without the Galerkin solve: it widens every bond and
    leaves the state unchanged, so a product start is not stuck at rank one.

    Attributes:
        warmup_sweeps: Passes before the step's own update, each doubling every bond where
            the neighbouring legs allow. Defaults to 0 (off); the RAGE integrators default to 1.
        warmup_every_step: ``None`` (the default) warms every step if ``time_dep``, else only
            the first. ``True``/``False`` force it.
    """
    warmup_sweeps: int = 0
    warmup_every_step: Optional[bool] = None


def warmup_passes(config, steps_done: int) -> int:
    """How many warm-up passes a step takes, from the config and the step count.

    Args:
        config: Any config carrying the :class:`WarmupConfig` fields.
        steps_done: Steps of the run already completed.

    Returns:
        ``config.warmup_sweeps`` when a warm-up applies at this step, otherwise 0.
    """
    n_passes = max(0, int(config.warmup_sweeps))
    if n_passes == 0 or steps_done == 0:
        return n_passes
    recurring = config.warmup_every_step
    if recurring is None:                       # a new generator per level
        recurring = bool(config.time_dep)
    return n_passes if recurring else 0


@dataclass
class ResidualSelectionConfig(WarmupConfig):
    """The residual-selection knobs, read by RAGE_MPS and RAGE_TTN.

    The selection threshold is ``rel_tol``, so a direction is added only if it would survive
    truncation.

    Attributes:
        max_aug_bond_dim: Safety cap on a bond's augmented dimension during the sweep.
            Infinity (the default) or ``None`` means no cap.
        residual_krylov_tol: Cutoff on the Krylov residual estimate, setting the two-site
            recurrence depth. ``0`` runs to the noise floor (long-range generators need it).
            Defaults to 1e-10.
        warmup_sweeps: Defaults to 1 here, so the first step has room to select from.
    """
    max_aug_bond_dim: Union[int, float] = float("inf")
    residual_krylov_tol: float = 1e-10
    warmup_sweeps: int = 1


@dataclass
class BUGConfig(ParameterBudgetConfig, WarmupConfig, IntegratorConfig):
    """Configuration for the recursive tree integrator ``BUG``.

    Attributes:
        kstep_evo_mode: Local solver for the K-step at every node. Defaults to
            ``TimeEvoMode.RK45`` at ``atol = rtol = 1e-10``; options in ``solver_options``.
        enable_parallel: Update sibling subtrees concurrently, with the same result as the
            serial path. Defaults to False.
        max_workers: Threads for ``enable_parallel``; more than the tree's branching factor
            does not help.
    """
    kstep_evo_mode: TimeEvoMode = TimeEvoMode.RK45
    solver_options: Dict[str, Any] = field(default_factory=dict)
    enable_parallel: bool = False
    max_workers: Optional[int] = field(default_factory=lambda: min(3, os.cpu_count() or 1))

    def local_solver(self) -> tuple[TimeEvoMode, Dict[str, Any]]:
        """The ``(mode, options)`` pair for this config's K-step solves."""
        return self._solver_pair(self.kstep_evo_mode)

    def __post_init__(self):
        """Validate the budget and warm-up knobs, then resolve the K-step options."""
        _validate_parameter_budget(self)
        _validate_warmup_knobs(self)
        self._resolve_solver_options(self.kstep_evo_mode, "kstep_evo_mode")
        super().__post_init__()


@dataclass
class BUGMPSConfig(SweepCompositionConfig, ParameterBudgetConfig, WarmupConfig,
                   IntegratorConfig):
    """Configuration for the chain integrator ``BUG_MPS``.

    Attributes:
        kstep_evo_mode: Local solver for the K-step at every site. Defaults to
            ``TimeEvoMode.RK45`` at ``atol = rtol = 1e-10``; options in ``solver_options``.
        composition: How one step is assembled from sweeps; see :class:`SweepComposition`.
            Defaults to ``DOUBLE_ENDPOINT``.
        warmup_sweeps: Warm-up sweeps without the chain-end Galerkin solve. Defaults to 0.
    """
    kstep_evo_mode: TimeEvoMode = TimeEvoMode.RK45
    solver_options: Dict[str, Any] = field(default_factory=dict)

    def local_solver(self) -> tuple[TimeEvoMode, Dict[str, Any]]:
        """The ``(mode, options)`` pair for this config's K-step solves."""
        return self._solver_pair(self.kstep_evo_mode)

    def __post_init__(self):
        """Validate the warm-up and budget, then resolve the options against the K-step."""
        _validate_warmup_knobs(self)
        _validate_parameter_budget(self)
        self._resolve_solver_options(self.kstep_evo_mode, "kstep_evo_mode")
        super().__post_init__()


@dataclass
class BUGTTNConfig(SweepCompositionConfig, ParameterBudgetConfig, IntegratorConfig):
    """Configuration for the sweeping tree integrator ``BUG_TTN``.

    Attributes:
        kstep_evo_mode: Local solver for the K-step at every node. Defaults to
            ``TimeEvoMode.RK45`` at ``atol = rtol = 1e-10``; options in ``solver_options``.
        composition: How one step is assembled from sweeps; see :class:`SweepComposition`.
            Defaults to ``DOUBLE_ENDPOINT``.
    """
    kstep_evo_mode: TimeEvoMode = TimeEvoMode.RK45
    solver_options: Dict[str, Any] = field(default_factory=dict)

    def local_solver(self) -> tuple[TimeEvoMode, Dict[str, Any]]:
        """The ``(mode, options)`` pair for this config's K-step solves."""
        return self._solver_pair(self.kstep_evo_mode)

    def __post_init__(self):
        """Validate the budget, then resolve the options against the K-step mode."""
        _validate_parameter_budget(self)
        self._resolve_solver_options(self.kstep_evo_mode, "kstep_evo_mode")
        super().__post_init__()


@dataclass
class RAGEMPSConfig(ResidualSelectionConfig, ParameterBudgetConfig, IntegratorConfig):
    """Configuration for the chain integrator RAGE_MPS: residual selection and segmentation.

    Attributes:
        max_segment_bond: Start a new segment once an augmented bond exceeds this. ``None``
            (the default), non-positive or infinity disable it. Typical: ``2 * max_bond_dim``.
        max_segment_width: Start a new segment once it spans this many sites. Lossless only
            from one light cone, ``ceil(2 e ||h|| dt) + 1``, up. Defaults to ``None``.
        galerkin_evo_mode: Local solver for the chain-end and segment Galerkin solves.
            Defaults to ``TimeEvoMode.KRYLOV``.
    """
    max_segment_bond: Optional[Union[int, float]] = None
    max_segment_width: Optional[Union[int, float]] = None
    galerkin_evo_mode: TimeEvoMode = TimeEvoMode.KRYLOV
    solver_options: Dict[str, Any] = field(default_factory=dict)

    def local_solver(self) -> tuple[TimeEvoMode, Dict[str, Any]]:
        """The ``(mode, options)`` pair for this config's Galerkin solves."""
        return self._solver_pair(self.galerkin_evo_mode)

    def __post_init__(self):
        """Validate the budget, warm-up, residual and segmentation knobs, then SVD."""
        _validate_parameter_budget(self)
        _validate_warmup_knobs(self)
        _validate_residual_knobs(self)
        _validate_segmentation_knobs(self)
        self._resolve_solver_options(self.galerkin_evo_mode,
                                     "galerkin_evo_mode")
        super().__post_init__()


@dataclass
class RAGETTNConfig(ResidualSelectionConfig, ParameterBudgetConfig, IntegratorConfig):
    """Configuration for the tree integrator RAGE_TTN: residual selection and node size.

    Attributes:
        max_node_size: Split any node whose augmented tensor would exceed this into virtual
            nodes (exact), down to three neighbour legs. Requires a static TTNO. ``None`` (the
            default), non-positive or infinity disable it.
        galerkin_evo_mode: Local solver for the root Galerkin solve. Defaults to
            ``TimeEvoMode.KRYLOV``.
    """
    max_node_size: Optional[Union[int, float]] = None
    galerkin_evo_mode: TimeEvoMode = TimeEvoMode.KRYLOV
    solver_options: Dict[str, Any] = field(default_factory=dict)

    def local_solver(self) -> tuple[TimeEvoMode, Dict[str, Any]]:
        """The ``(mode, options)`` pair for this config's Galerkin solves."""
        return self._solver_pair(self.galerkin_evo_mode)

    def __post_init__(self):
        """Validate the budget, warm-up, residual and node-size knobs, then SVD."""
        _validate_parameter_budget(self)
        _validate_warmup_knobs(self)
        _validate_residual_knobs(self)
        _validate_node_size(self)
        self._resolve_solver_options(self.galerkin_evo_mode,
                                     "galerkin_evo_mode")
        super().__post_init__()


def basis_change_tensor_id(node_id: str) -> str:
    """Identifier of the basis-change node the recursive BUG interposes at ``node_id``."""
    return node_id + "_basis_change_tensor"


def reverse_basis_change_tensor_id(node_id: str) -> str:
    """Inverse of basis_change_tensor_id: the node identifier it was built from."""
    return node_id[:-len("_basis_change_tensor")]


def _compute_new_basis_tensor_qr(node: Node,
                                 combined_tensor: ndarray,
                                 neighbour_id: Optional[str] = None) -> ndarray:
    """The Q factor of ``combined_tensor`` across the bond to ``neighbour_id``, in the node's
    own leg order.

    Args:
        node: The node being updated; supplies the reference leg order.
        combined_tensor: A tensor in the node's leg order -- for a basis update, the
            ``[old | evolved]`` pair concatenated along the neighbour leg.
        neighbour_id: The neighbour the bond points at. Defaults to the node's parent.

    Returns:
        The new basis tensor ``Q``, in the node's leg order.
    """
    if neighbour_id is None:
        neighbour_id = node.parent
    q_legs, r_legs = _build_leg_specs(node, neighbour_id)
    q_legs.node = node
    r_legs.node = node
    q_axes = q_legs.find_leg_values()
    new_basis_tensor, _ = tensor_qr_decomposition(combined_tensor, q_axes,
                                                  r_legs.find_leg_values())
    # QR position of each original leg; the new bond sits last in the QR output.
    qr_position = {leg: i for i, leg in enumerate(q_axes)}
    qr_position[node.neighbour_index(neighbour_id)] = len(q_axes)
    node_order = [] if node.is_root() else [node.parent_leg]
    node_order += [node.neighbour_index(child_id) for child_id in node.children]
    node_order += list(node.open_legs)
    return transpose(new_basis_tensor, [qr_position[leg] for leg in node_order])


def compute_new_basis_tensor(node: Node,
                             old_tensor: ndarray,
                             updated_tensor: ndarray,
                             neighbour_id: Optional[str] = None) -> ndarray:
    """An orthonormal basis of the ``[old | evolved]`` range (QR), in the node's leg order.

    Args:
        node: The node being updated.
        old_tensor: Its current tensor, in the node's leg order. Its node must be the
            orthogonality centre of the old state.
        updated_tensor: The time-evolved tensor, in the same leg order.
        neighbour_id: The neighbour holding the orthogonality centre. Defaults to the
            node's parent.

    Returns:
        The new basis tensor, in the node's leg order, with a bond toward the neighbour
        of at most twice the old dimension.
    """
    if neighbour_id is None:
        neighbour_id = node.parent
    combined_tensor = concatenate((old_tensor, updated_tensor),
                                  axis=node.neighbour_index(neighbour_id))
    return _compute_new_basis_tensor_qr(node, combined_tensor, neighbour_id)


def compute_basis_change_tensor(node_old: Node,
                                node_new: Node,
                                tensor_old: ndarray,
                                tensor_new: ndarray,
                                basis_change_tensor_cache: PartialTreeCachDict
                                ) -> ndarray:
    """The basis change ``M = old^dagger new``, using the children's cached basis changes.

    Args:
        node_old: Node of the old basis tensor. Must not be the root.
        node_new: Node of the new basis tensor.
        tensor_old: The old basis tensor.
        tensor_new: The new basis tensor.
        basis_change_tensor_cache: Basis-change tensors of the already-updated children.

    Returns:
        A matrix whose first leg carries the old bond dimension and second the new one.
    """
    parent_id = node_old.parent
    if parent_id is None:
        raise ValueError("The root node has no parent, so it has no basis change.")
    return contract_any_nodes(parent_id, node_old, node_new,
                              tensor_old, tensor_new.conj(),
                              basis_change_tensor_cache)


def adjust_node1_structure_to_node2(start_ttn, target_ttn, node_id: str) -> None:
    """Permute one node's legs and children to match the same node in ``target_ttn``.

    A no-op when the two already agree.

    Args:
        start_ttn: The TTN to adjust, in place.
        target_ttn: The TTN supplying the reference leg order.
        node_id: Identifier of the node to adjust.

    Raises:
        NotImplementedError: The node carries more than two open legs.
    """
    start_node = start_ttn.nodes[node_id]
    target_node = target_ttn.nodes[node_id]
    legs = get_equivalent_legs(start_node, target_node)
    if legs[0] == legs[1]:
        return
    start_neighbours = start_node.neighbouring_nodes()
    position = {neighbour: i for i, neighbour in enumerate(start_neighbours)}
    permutation = tuple(position[neighbour]
                        for neighbour in target_node.neighbouring_nodes())
    nneighbours = target_node.nneighbours()
    n_open = len(start_node.open_legs)
    if n_open > 2:
        raise NotImplementedError(
            f"Node {node_id} has {n_open} open legs; at most two are supported.")
    permutation += tuple(nneighbours + i for i in range(n_open))
    start_node.update_leg_permutation(permutation, start_ttn.tensors[node_id].shape)
    start_node.children = target_node.children.copy()


def adjust_ttn1_structure_to_ttn2(start_ttn, target_ttn) -> None:
    """Bring every node of ``start_ttn`` into ``target_ttn``'s leg order.

    Args:
        start_ttn: The TTN to adjust, in place.
        target_ttn: The TTN supplying the reference leg order.
    """
    for node_id in start_ttn.nodes:
        adjust_node1_structure_to_node2(start_ttn, target_ttn, node_id)
