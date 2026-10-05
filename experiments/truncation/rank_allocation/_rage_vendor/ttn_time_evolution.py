"""
This module implements an abstract time evolution for tree tensor networks.

On top of the driver it inherits, it records the per-step diagnostics: bond dimensions,
state size, norm and Loschmidt amplitude, each behind its own ``record_*`` config flag.
"""

from __future__ import annotations
from typing import List, Union, Dict, Tuple
from dataclasses import dataclass

from pytreenet.time_evolution.time_evolution import TimeEvolution, TimeEvoConfig
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.ttno import TTNO
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.ttno.time_dep_ttno import AbstractTimeDepTTNO


__all__ = [
    "TTNTimeEvolutionConfig", "TTNTimeEvolution", "TTNOBasedTimeEvolution",
    "MAX_BOND_DIM_ID", "AVERAGE_BOND_DIM_ID", "TOTAL_SIZE_ID", "BOND_DIM_ATTR"
]


MAX_BOND_DIM_ID = "max_bond_dim"
AVERAGE_BOND_DIM_ID = "average_bond_dim"
TOTAL_SIZE_ID = "total_size"
BOND_DIM_ATTR = "bond_dim"

@dataclass
class TTNTimeEvolutionConfig(TimeEvoConfig):
    """
    Configuration for the TTN time evolution.

    Attributes:
        record_bond_dim: Record every bond's dimension, one result key per bond.
        record_max_bdim: Record the largest bond dimension of the state.
        record_average_bdim: Record the mean bond dimension of the state.
        record_total_size: Record the total number of tensor entries in the state.
        record_norm: Record the norm of the state.
        record_loschmidt_amplitude: Record the overlap with the initial state.

    Each is evaluated after every step at which observables are evaluated, and each
    defaults to False.
    """
    record_bond_dim: bool = False
    record_max_bdim: bool = False
    record_average_bdim: bool = False
    record_total_size: bool = False
    record_norm: bool = False
    record_loschmidt_amplitude: bool = False

class TTNTimeEvolution(TimeEvolution):
    """
    A time evolution for tree tensor networks.

    Provides functionality to compute expectation values of operators during
    the time evolution and record bond dimensions of the current state.
    """
    config_class = TTNTimeEvolutionConfig

    def __init__(self, initial_state: TreeTensorNetworkState,
                 time_step_size: float, final_time: float,
                 operators: Union[List[Union[TensorProduct, TTNO]],
                                  Dict[str, Union[TensorProduct, TTNO]],
                                  TensorProduct,
                                  TTNO],
                 config: Union[TTNTimeEvolutionConfig,None] = None
                 ) -> None:
        """
        A time evolution for a tree tensor network state.

        Args:
            initial_state: The initial state of the time
                evolution.
            time_step_size: The time difference progressed by one time
                step.
            final_time: The final time until which the time evolution
                runs.
            operators:
                Operators for which the expectation value should be recorded
                during the time evolution.
            config: The configuration of
                time evolution. Defaults to None.
        """
        super().__init__(initial_state, time_step_size, final_time, operators,
                         config=config,)
        self._initial_state: TreeTensorNetworkState
        self.state: TreeTensorNetworkState

        # Convert initial state entries to complex
        self.state.to_complex(128)
        self._initial_state.to_complex(128)

    def result_init_dictionary(self):
        """
        Initializes the result dictionary for the time evolution.

        This method adds entries for the bond dimension recording.
        """
        diction = super().result_init_dictionary()
        if self.config.record_bond_dim:
            bond_dims = self.obtain_bond_dims()
            for key in bond_dims.keys():
                diction[key] = int
                self.results.set_attribute(key, BOND_DIM_ATTR, True)
        if self.config.record_max_bdim:
            diction[MAX_BOND_DIM_ID] = int
        if self.config.record_average_bdim:
            diction[AVERAGE_BOND_DIM_ID] = float
        if self.config.record_total_size:
            diction[TOTAL_SIZE_ID] = int
        if self.config.record_norm:
            diction["norm"] = complex
        if self.config.record_loschmidt_amplitude:
            diction["loschmidt_amplitude"] = complex
        return diction

    def obtain_bond_dims(self) -> Dict[Tuple[str,str], int]:
        """
        Obtains a dictionary of all bond dimensions in the current state.
        """
        return self.state.bond_dims()

    def obtain_max_bond_dim(self) -> int:
        """
        Obtain the largest bond dimension of the current state.
        """
        return self.state.max_bond_dim()

    def obtain_average_bond_dim(self) -> float:
        """
        Obtain the average bond dimension of the current state.

        Returns:
            float: The average bond dimension of the current state.
        """
        return self.state.avg_bond_dim()

    def obtain_total_size(self) -> int:
        """
        Obtain the total size of the current state.

        Returns:
            int: The total size of the current state.
        """
        return self.state.size()

    def record_bond_dimensions(self, index):
        """
        Records the bond dimensions of the current state.

        This method is called after each time step to record the bond
        dimensions of the current state if recording is enabled.

        Args:
            index: The index at which to record the bond dimensions.

        """
        if self.config.record_bond_dim:
            for key, value in self.obtain_bond_dims().items():
                self.results.set_element(key, index, value)
        if self.config.record_max_bdim:
            max_bond_dim = self.obtain_max_bond_dim()
            self.results.set_element(MAX_BOND_DIM_ID, index, max_bond_dim)
        if self.config.record_average_bdim:
            average_bond_dim = self.obtain_average_bond_dim()
            self.results.set_element(AVERAGE_BOND_DIM_ID, index, average_bond_dim)
        if self.config.record_total_size:
            total_size = self.obtain_total_size()
            self.results.set_element(TOTAL_SIZE_ID, index, total_size)

    def obtain_loschmidt_amplitude(self) -> complex:
        """
        Obtain the Loschmidt amplitude of the current state.

        Returns:
            complex: The overlap of the current state with the initial state.
        """
        init = self.initial_state
        lo_amp = self.state.scalar_product(other=init)
        return lo_amp

    def evaluate_operators(self, index: int):
        """
        Evaluates the operator including the recording of bond dimensions.

        Args:
            index: The index at which to evaluate the operators.
        """
        super().evaluate_operators(index)
        self.record_bond_dimensions(index)
        if self.config.record_norm:
            norm = self.state.norm()
            self.results.set_element("norm", index, norm)
        if self.config.record_loschmidt_amplitude:
            los_amp = self.obtain_loschmidt_amplitude()
            self.results.set_element("loschmidt_amplitude",
                                     index,
                                     los_amp)

    def evaluate_operator(self, operator: Union[TensorProduct,TTNO]) -> complex:
        """
        Evaluate the expectation value of a single operator.

        Args:
            operator: The operator for which to compute the
                expectation value.

        Returns:
            complex: The expectation value of the operator with respect to
                the current state.
        """
        return self.state.operator_expectation_value(operator)

    def reset_hamiltonian(self):
        """Reset a time-dependent Hamiltonian to its initial layer.

        Defined here so every subclass holding a ``hamiltonian`` inherits it; only
        meaningful when that Hamiltonian is an AbstractTimeDepTTNO.
        """
        self.hamiltonian.reset()

class TTNOBasedTimeEvolution(TTNTimeEvolution):
    """
    A time evolution for tree tensor networks based on TTNOs.

    This class is used for time evolution algorithms that are based on
    tree tensor network operators (TTNOs). It provides additional methods
    for updating the Hamiltonian and resetting it to its initial state.
    """

    def __init__(self,
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TreeTensorNetworkOperator,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[List[Union[TensorProduct, TTNO]],
                                  Dict[str, Union[TensorProduct, TTNO]],
                                  TensorProduct,
                                  TTNO],
                 config: Union[TTNTimeEvolutionConfig, None] = None,
                 ) -> None:
        """
        Initializes the TTNOBasedTimeEvolution class.

        Args:
            initial_state: The initial state of the
                time evolution.
            hamiltonian: The Hamiltonian of the
                system.
            time_step_size: The time difference progressed by one time
                step.
            final_time: The final time until which the time evolution
                runs.
            operators:
                Operators for which the expectation value should be recorded
                during the time evolution.
            config: The configuration of
                time evolution. Defaults to None.
        """
        super().__init__(initial_state=initial_state,
                         time_step_size=time_step_size,
                         final_time=final_time,
                         operators=operators,
                         config=config)
        if self.config.time_dep and not isinstance(hamiltonian, AbstractTimeDepTTNO):
            errstr = "The Hamiltonian must be from the AbstractTimeDepTTNO class " \
                     "if the time evolution is time-dependent!"
            raise TypeError(errstr)
        self.hamiltonian = hamiltonian

    def update_hamiltonian(self):
        """
        Updates the Hamiltonian for the next time step and applies any state
        modification the completed layer carries (e.g. a reset/measurement from a
        DiscreetTimeTTNO circuit). Only required for time-dependent Hamiltonians.
        """
        self.hamiltonian.update(self.time_step_size)
        self.hamiltonian.modify_state(self.state)
