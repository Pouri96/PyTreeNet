"""
This module provides the abstract TimeEvolution class
"""
from __future__ import annotations
from typing import List, Union, Any, Dict, Iterable, Callable, Literal
from enum import Enum
from copy import deepcopy
from math import modf
from warnings import warn
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from tqdm import tqdm
from numpy.typing import DTypeLike

from ..util.ttn_exceptions import positivity_check, non_negativity_check
from ..util import fast_exp_action
from .results import Results

@dataclass
class TimeEvoConfig:
    """
    Abstract configuration for time evolution algorithms.
    
    This is to ensure that all time evolution algorithms can be configured.
    """
    time_dep: bool = False

class TimeEvolution:
    """
    An abstract class that can be used for various time-evolution algorithms.

    Contains all methods required to cover common discrete time evolutions and
    avoid code duplication. The algorithms runs a number of time steps of given
    size until a specified final time is reached.

    While results can be called directly as a big martix using the `results`
    attribute, there are convenience functions to easily extract the desired
    results.

    Attributes:
        initial_state (Any): The initial state of the system to be time evolved.
        time_step_size (float): The size of each discreet time step.
        final_time (float): The time at which to conclude the time-evolution.
        operators (List[Any]): A list of operators to be evaluated.
    """
    config_class = TimeEvoConfig

    def __init__(self,
                 initial_state: Any,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[List[Any], Dict[str, Any], Any],
                 solver_options: Union[Dict[str, Any], None] = None,
                 config: Union[TimeEvoConfig, None] = None):
        """
        Initialises a TimeEvolution object.

        Args:
            initial_state (Any): The initial state.
            time_step_size (float): The size of one time step.
            final_time (float): The final time.
            operators (Union[List[Any], Dict[str, Any], Any]): The operators
                for which to compute the expectation values. Can be a single
                operator or a list of operators. If a dictionary is given, the
                results can be called by using the keys of the dictionary.
            solver_options (Union[Dict[str, Any], None], optional): Most time
                evolutions algorithms use some kind of solver to resolve a
                partial differential equation. This dictionary can be used to
                pass additional options to the solver. Refer to the
                documentation of `ptn.time_evolution.TimeEvoMode` for further
                information. Defaults to None.
            config (Union[TimeEvoConfig, None], optional): The configuration
                for the time evolution algorithm. If None, a default
                configuration is used.
        """
        # Both attributes are private copies: the solver must never mutate the
        # caller's state object. The working copy is upcast/re-gauged in place
        # by subclasses; _initial_state additionally receives in-place
        # structure adjustments (e.g. BUG_TTN's adjust_ttn1_structure_to_ttn2)
        # and the complex upcast, none of which may leak to the caller: an
        # aliased _initial_state would rewrite a real state's tensors to
        # complex in place.
        self._initial_state = deepcopy(initial_state)
        self.state = deepcopy(initial_state)
        positivity_check(time_step_size, "size of one time step")
        self._time_step_size = time_step_size
        positivity_check(final_time, "final time")
        self._final_time = final_time
        self._num_time_steps = self._compute_num_time_steps()
        self.operators = self.init_operators(operators)
        if config is None:
            self.config = self.config_class()
        else:
            self.config = config
        if solver_options is None:
            self.solver_options = {}
        else:
            self.solver_options = solver_options
        self.results = Results()

    def _compute_num_time_steps(self) -> int:
        """
        Compute the number of time steps from attributes.

        If the decimal part of the time steps is close to 0, the calculated
        number of time steps is directly returned. Otherwise, it is assumed
        to be better to run one more time step.
        """
        decimal, integer = modf(self._final_time / self._time_step_size)
        threshold = 0.1
        if decimal < threshold:
            return int(integer)
        return int(integer + 1)

    def init_operators(self,
                       operators: Union[List[Any], Dict[str, Any], Any]
                       ) -> Dict[str, Any]:
        """
        Initialises the operators for which to compute the expectation values.

        Args:
            operators (Union[List[Any], Dict[str, Any], Any]): The operators
                for which to compute the expectation values. Can be a single
                operator or a list of operators. If a dictionary is given, the
                results can be called by using the keys of the dictionary
                otherwise the keys are the indices of the operators in the list.

        Returns:
            Dict[str, Any]: A dictionary mapping the operator names to the
                operators. If a list of operators is given, the keys are the
        """
        if isinstance(operators, List):
            return {str(i): op for i, op in enumerate(operators)}
        if isinstance(operators, Dict):
            return operators
        return {"0": operators}

    @property
    def initial_state(self) -> Any:
        """
        Returns the initial state.
        """
        return self._initial_state

    @property
    def time_step_size(self) -> float:
        """
        Returns the size of one time step.
        """
        return self._time_step_size

    @property
    def final_time(self) -> float:
        """
        Returns the final time.
        """
        return self._final_time

    @property
    def num_time_steps(self) -> int:
        """
        Returns the current number of time steps.
        """
        return self._num_time_steps

    def set_num_time_steps(self, num_time_steps: int):
        """
        Set the number of time steps to be run.
        
        Sometimes it is more convenient to define the size of the time steps
        and the number of steps to be run, rather than using a final time.
        This method modifies the internal attributes accordingly.
        """
        non_negativity_check(num_time_steps, "number of time steps")
        self._num_time_steps = num_time_steps
        self._final_time = num_time_steps * self._time_step_size

    def set_num_time_steps_constant_final_time(self, num_time_steps: int):
        """
        Sets the number of time-steps and keeps the final time constant.
        
        The internal attributes are modified accordingly.
        """
        non_negativity_check(num_time_steps, "number of time steps")
        self._num_time_steps = num_time_steps
        self._time_step_size = self._final_time / num_time_steps

    def run_one_time_step(self, **kwargs):
        """
        Abstract method to run one time step.
        """
        raise NotImplementedError()

    def evaluate_operator(self, operator: Any) -> complex:
        """
        Abstract method to evaluate the expectation value of a single operator.

        Args:
            operator (Any): The operator for which to compute the expectation
                value.

        Returns:
            complex: The expectation value of the operator.
        """
        raise NotImplementedError()

    def update_hamiltonian(self):
        """
        Abstract method to update the Hamiltonian for the next time step.
        
        This is only required for time-dependent Hamiltonians.
        """
        raise NotImplementedError("This method should be implemented in a subclass!")

    def evaluate_operators(self, index: int):
        """
        Evaluate the expectation value for all operators for the current state.

        Args:
            index (int): The index at which to save the results.
        """
        for key, operator in self.operators.items():
            exp_val = self.evaluate_operator(operator)
            self.results.set_element(key, index, exp_val)

    def result_init_dictionary(self) -> Dict[str, DTypeLike]:
        """
        Returns a dictionary with the initialisation of the results.

        The keys are the operator names and the values are the data types
        of the results.
        
        Returns:
            Dict[str, DTypeLike]: The initialisation dictionary for the results.
        """
        return {key: complex for key in self.operators.keys()}

    def init_results(self, evaluation_time: Union[int,Literal["inf"]] = 1):
        """
        Initialises an appropriately sized zero valued numpy array for storage.

        Each row contains the results obtained for one operator, while the
        last row contains the times. Note, the the entry with index zero
        corresponds to time 0.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
                Defaults to 1.
        """
        if evaluation_time == "inf":
            num_results = 1
        else:
            num_results = self.num_time_steps // evaluation_time
        res_dtypes = self.result_init_dictionary()
        self.results.initialize(res_dtypes, num_results)

    def save_time(self, time_step: int, index: int):
        """
        Saves the current time at a given index.

        Args:
            time_step (int): The current time step.
            index (int): The index at which to save the time.

        """
        self.results.set_time(index, time_step*self.time_step_size)

    def result_index(self,
                     evalutation_time: Union[int,Literal["inf"]],
                     time_step: int) -> int:
        """
        Returns the time index at which to save the results.

        Args:
            evalutation_time (Union[int,Literal["inf"]]): The difference in time steps
                after which to evaluate the operator expectation values.
            time_step (int): The current time step.
        
        Returns:
            int: The index at which to save the results.

        """
        if evalutation_time == "inf":
            return 0
        return time_step // evalutation_time

    def should_evaluate(self,
                        evaluation_time: Union[int,Literal["inf"]],
                        time_step: int
                        ) -> bool:
        """
        Returns if the operators should be evaluated at the current time step.

        Args:
            evaluation_time (Union[int,Literal["inf"]]): The difference in time steps after which
                to evaluate the operator expectation values.
            time_step (int): The current time step.
        
        Returns:
            bool: If the operators should be evaluated at the current time step.
        
        """
        some_time_step = evaluation_time != "inf" and time_step % evaluation_time == 0
        end_time = evaluation_time == "inf" and time_step == self.num_time_steps
        return some_time_step or end_time

    def evaluate_and_save_results(self,
                                  evaluation_time: Union[int,Literal["inf"]],
                                  time_step: int):
        """
        Evaluates and saves the operator results at a given time step.

        Args:
            evaluation_time (Union[int,Literal["inf"]]): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
            time_step (int): The current time step.

        """
        if self.should_evaluate(evaluation_time, time_step):
            index = self.result_index(evaluation_time, time_step)
            self.evaluate_operators(index)
            self.save_time(time_step, index)

    def create_run_tqdm(self, pgbar: bool = True) -> Iterable:
        """
        Creates the decorated iterator for the progress bar.

        Args:
            pgbar (bool, optional): If True, a progress bar is shown. 
                Defaults to True.
        
        Returns:
            Iterable: The decorated iterator.

        """
        return tqdm(range(self.num_time_steps + 1), disable=not pgbar)

    def run(self,
            evaluation_time: Union[int,Literal["inf"]] = 1,
            pgbar: bool = True):
        """
        Runs this time evolution algorithm for the given parameters.

        The desired operator expectation values are evaluated and saved.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
                Defaults to 1.
            pgbar (bool, optional): Toggles the progress bar. Defaults to True.
        """
        self.init_results(evaluation_time)
        for i in self.create_run_tqdm(pgbar):
            if i != 0:  # We also measure the initial expectation_values
                self.run_one_time_step()
            self.evaluate_and_save_results(evaluation_time, i)
            if self.config.time_dep and i != 0:
                self.update_hamiltonian()

    def reset_hamiltonian(self):
        """
        Resets the Hamiltonian to its initial state.

        This is only required for time-dependent Hamiltonians.
        """
        raise NotImplementedError("This method should be implemented in a subclass!")

    def reset_to_initial_state(self):
        """
        Resets the current state to the intial state
        """
        self.state = deepcopy(self._initial_state)
        self.results = Results()
        # The INSTANCE's flag, not the class default: ``config_class.time_dep`` reads
        # the dataclass default, so a caller who set time_dep=True on the config it
        # passed in got no Hamiltonian reset here.
        if self.config.time_dep:
            self.reset_hamiltonian()

class EvoDirection(Enum):
    """
    Enum for the direction of time evolution.
    """
    FORWARD = -1
    BACKWARD = 1

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_bool(forward: bool) -> EvoDirection:
        """
        Converts a boolean to a EvoDirection enum.

        Args:
            forward (bool): If True, FORWARD is returned. Otherwise BACKWARD.

        Returns:
            EvoDirection: The corresponding EvoDirection enum.
        """
        return EvoDirection.FORWARD if forward else EvoDirection.BACKWARD

    def exp_sign(self) -> int:
        """
        Returns the sign of the exponent for the time evolution.

        Returns:
            int: The sign of the exponent.
        """
        return self.value

    def exp_factor(self) -> complex:
        """
        Returns the factor for the exponent in the time evolution.

        Returns:
            complex: The factor for the exponent.
        """
        return 1.0j * self.exp_sign()

class TimeEvoMode(Enum):
    """
    Mode for the time evolution of a matrix.
    """

    FASTEST = "fastest"
    EXPM = "expm"
    CHEBYSHEV = "chebyshev"
    SPARSE = "sparse"
    RK45 = "RK45"
    RK23 = "RK23"
    DOP853 = "DOP853"
    BDF = "BDF"
    KRYLOV = "krylov"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def default_solver_options(mode: TimeEvoMode) -> Dict[str, Any]:
        """
        The solver options a mode uses when the caller supplies none.

        The scipy solvers are the reason this exists: left to their own defaults
        ``solve_ivp`` integrates at ``rtol=1e-3``, which is far looser than any use
        here and fails silently -- it returns a plausible trajectory rather than an
        error. The matrix-exponential modes take no options.

        Args:
            mode (TimeEvoMode): The mode whose defaults are wanted.

        Returns:
            Dict[str, Any]: A fresh options dictionary, safe for the caller to mutate.
        """
        if mode in TimeEvoMode.scipy_modes():
            return {'atol': 1e-10, 'rtol': 1e-10}
        if mode is TimeEvoMode.KRYLOV:
            return {'krylov_tol': 1e-10}
        return {}

    @staticmethod
    def valid_solver_options(mode: TimeEvoMode) -> set[str]:
        """
        The option keys a mode actually reads.

        Anything outside this set is silently dropped by the solver it is handed to
        -- ``krylov_exp_action`` ignores what it does not pop, and the matrix
        exponential modes take no options at all -- so callers use this to reject a
        mismatch (``atol`` on a KRYLOV mode, say) rather than run at a tolerance that
        was never applied.

        Args:
            mode (TimeEvoMode): The mode whose option keys are wanted.

        Returns:
            set[str]: The accepted keys. ``exp_factor`` and ``hermitian`` are
                accepted by every action-evolvable mode.
        """
        if mode is TimeEvoMode.KRYLOV:
            return {'krylov_tol', 'max_krylov_dim', 'hermitian', 'exp_factor'}
        if mode in TimeEvoMode.scipy_modes():
            return {'atol', 'rtol', 'max_step', 'first_step', 'jac', 'jac_sparsity',
                    'lband', 'uband', 'min_step', 'hermitian', 'exp_factor'}
        return {'exp_factor'}

    @staticmethod
    def fastest_equivalent() -> TimeEvoMode:
        """
        Selects the mode that is equivalent to the fastest.
        """
        return TimeEvoMode.CHEBYSHEV

    @staticmethod
    def scipy_modes() -> List[TimeEvoMode]:
        """
        Returns a list of all modes that are scipy ODE solvers.
        
        Returns:
            List[TimeEvoMode]: The list of scipy modes.
        """
        return [TimeEvoMode.RK45,
                TimeEvoMode.RK23,
                TimeEvoMode.DOP853,
                TimeEvoMode.BDF]

    def is_scipy(self) -> bool:
        """
        Determines, if this mode is a scipy ODE solver.
        """
        if self == TimeEvoMode.FASTEST:
            return self.fastest_equivalent().is_scipy()
        return self in self.scipy_modes()

    def action_evolvable(self) -> bool:
        """
        Determines, if this mode can be used for time evolution via an action.

        Returns:
            bool: True if the mode is suitable for time evolution via an action.
        """
        return self.is_scipy() or self is TimeEvoMode.KRYLOV

    def time_evolve_action(self,
                           psi: np.ndarray,
                           time_evo_action: Callable,
                           time_difference: float,
                           forward: EvoDirection| bool = EvoDirection.FORWARD,
                           **options
                           ) -> np.ndarray:
        """
        Performs the time evolution of a tensor from an action.

        Args:
            psi (np.ndarray): The initial state as an arbitrary tensor.
            time_evo_action (Callable): The action to be performed on the
                right hand side of the effective Schrödinger equation. Takes
                the time and the state as arguments.
            time_difference (float): The duration of the time-evolution.
            forward (EvoDirection|bool, optional): The direction of the time evolution.
            **options: Additional options for the solver underlying the time
                evolution. See the scipy documentation for the available options of
                the specific solver, and ``valid_solver_options`` for the keys each
                mode reads. Two are handled here rather than passed on:
                ``exp_factor`` and ``hermitian``.

        Returns:
            np.ndarray: The time-evolved state.
        """
        if isinstance(forward, bool):
            forward = EvoDirection.from_bool(forward)
        # ``exp_factor`` overrides the -/+ i of the Schroedinger convention, so the same
        # machinery can integrate y' = f A y for a real f -- which keeps the state, the
        # Krylov basis and the SVDs in real arithmetic when the generator is real.
        #
        # The override gives the FORWARD factor; the backward factor is its negation. A
        # direction-blind constant is not admissible: a segment collapse evolves forward
        # at the checkpoint and then backward to un-rotate, and a constant factor makes
        # the second step repeat the first instead of inverting it -- silently, and only
        # when segmentation is active. Consistent with the default, since
        # forward.exp_factor() = -/+1j already flips sign with the direction; omitting
        # the option is bit-for-bit the unoverridden behaviour.
        _override = options.pop('exp_factor', None)
        exp_factor = (forward.exp_factor() if _override is None
                      else (_override if forward is EvoDirection.FORWARD else -_override))
        real_flow = not np.iscomplexobj(np.asarray(exp_factor))
        # Hermiticity of the generator (the action). Popped here so it never reaches
        # solve_ivp on the scipy branch; only the KRYLOV path is specialised for it
        # (EXPM/RK45 handle a non-Hermitian generator natively).
        hermitian = options.pop('hermitian', True)
        if self is TimeEvoMode.KRYLOV:
            # Matrix-free Krylov (Lanczos) exponential: exact for the linear ODE
            # y' = exp_factor * A y, i.e. exp(exp_factor * dt * A) psi, without ever
            # forming A. With hermitian=True the small propagator uses eigh; with
            # hermitian=False, the general exp of the raw Arnoldi projection.
            # A real flow on a real state stays real; upcasting would throw the win away.
            if not (real_flow or np.issubdtype(psi.dtype, np.complexfloating)):
                psi = psi.astype(np.complex128)
            krylov_tol = options.pop('krylov_tol', 1e-10)
            max_krylov_dim = options.pop('max_krylov_dim', None)
            return krylov_exp_action(psi,
                                     lambda y: time_evo_action(0.0, y),
                                     time_difference, exp_factor,
                                     krylov_tol=krylov_tol,
                                     max_krylov_dim=max_krylov_dim,
                                     hermitian=hermitian)
        if self.is_scipy():
            orig_shape = psi.shape
            if not (real_flow or np.issubdtype(psi.dtype, np.complexfloating)):
                psi = psi.astype(np.complex128)
                warn("A real tensor was supplied for a complex flow; it is cast to "
                     "complex128.", UserWarning, stacklevel=2)
            def ode_rhs(t, y_vec):
                orig_array = y_vec.reshape(orig_shape)
                action_result = time_evo_action(t, orig_array).flatten()
                return exp_factor * action_result
            t_span = (0, time_difference)
            res = solve_ivp(ode_rhs, t_span, psi.flatten(),
                            method=self.value,
                            t_eval=[time_difference],
                            **options)
            return res.y[:, 0].reshape(orig_shape)
        raise NotImplementedError(
            "The time evolution via action is not implemented for this mode: "
            + str(self.value)
        )

    def time_evolve(self,
                    psi: np.ndarray,
                    hamiltonian: np.ndarray,
                    time_difference: float,
                    forward: EvoDirection| bool = EvoDirection.FORWARD,
                    **options: dict[str, Any]) -> np.ndarray:
        """
        Time evolves a state psi via a Hamiltonian matrix.

        Args:
            psi (np.ndarray): The initial state as a vector.
            hamiltonian (np.ndarray): The Hamiltonian determining the dynamics as
                a matrix.
            time_difference (float): The duration of the time-evolution
            forward (EvoDirection|bool, optional): The direction of the time evolution.
                    Defaults to EvoDirection.FORWARD.
            **options (dict[str, Any]): Additional options for the solver
                underlying the time evolution. See the scipy documentation for
                the available options of the specific solver.
        
        Returns:
            np.ndarray: The time evolved state
        """
        if isinstance(forward, bool):
            forward = EvoDirection.from_bool(forward)
        if self.action_evolvable():
            return self.time_evolve_action(psi,
                                           lambda t, y: hamiltonian @ y.flatten(),
                                           time_difference,
                                           forward=forward,
                                           **options)
        # This branch does not route through time_evolve_action, so an ``exp_factor``
        # carried in the options would otherwise be dropped here.
        _override = options.pop('exp_factor', None)      # forward factor; backward negates it
        exp_factor = (forward.exp_factor() if _override is None
                      else (_override if forward is EvoDirection.FORWARD else -_override))
        exponent = exp_factor * hamiltonian * time_difference
        return fast_exp_action(exponent, psi.flatten(),
                               mode=self.value).reshape(psi.shape)

def time_evolve(psi: np.ndarray, hamiltonian: np.ndarray,
                time_difference: float,
                forward: EvoDirection| bool = EvoDirection.FORWARD,
                mode: TimeEvoMode = TimeEvoMode.FASTEST) -> np.ndarray:
    """
    Time evolves a state psi via a Hamiltonian.
     
    The evolution can be either forward or backward in time:
        psi(t +/- dt) = exp(-/+ i*h*dt) @ psi(t)
        -iHdt: forward = True
        +iHdt: forward = False

    Args:
        psi (np.ndarray): The initial state as a vector.
        hamiltonian (np.ndarray): The Hamiltonian determining the dynamics as
            a matrix.
        time_difference (float): The duration of the time-evolution
        forward (EvoDirection|bool, optional): The direction of the time evolution.
                Defaults to EvoDirection.FORWARD.
        mode (TimeEvoMode, optional): The mode to use for the time evolution.

    Returns:
        np.ndarray: The time evolved state
    """
    if isinstance(forward, bool):
        forward = EvoDirection.from_bool(forward)
    sign = forward.exp_sign()
    rhs_matrix = sign * 1.0j * hamiltonian
    if mode is TimeEvoMode.KRYLOV:
        return mode.time_evolve(psi, hamiltonian, time_difference, forward=forward)
    if mode.is_scipy():
        def ode_rhs(_, y_vec):
            return rhs_matrix @ y_vec
        t_span = (0, time_difference)
        solution = solve_ivp(ode_rhs, t_span, psi.flatten(),
                                method=mode.value,
                                t_eval=[time_difference])
        result_vector = solution.y[:,0]
    else:
        exponent = rhs_matrix * time_difference
        result_vector = fast_exp_action(exponent, psi.flatten(),
                                        mode=mode.value)
    return result_vector.reshape(
                      psi.shape)


def _krylov_small_propagator(t_mat: np.ndarray,
                             time_difference: float,
                             exp_factor: complex,
                             hermitian: bool = True) -> np.ndarray:
    """
    First column ``exp(exp_factor * dt * T) e_1`` of the small Krylov propagator.

    ``T`` is the Arnoldi projection of the action onto the Krylov subspace. When the
    generator is Hermitian (``hermitian=True``, the default) ``T`` is Hermitian up to
    rounding, so the exponential is taken from the eigendecomposition of the
    symmetrised matrix (``eigh``): exact for the linear flow, cheap for the tiny
    ``m x m`` matrices built here, and the source of the unit-modulus (norm-preserving)
    real-time coefficients. For a NON-Hermitian generator (``hermitian=False``) ``T``
    is a genuine upper-Hessenberg matrix whose anti-Hermitian part the symmetrisation
    would discard, so the general dense matrix exponential of the raw ``T`` is used
    instead. Only the first column is needed -- for both the propagated vector and the
    a-posteriori residual estimate.
    """
    if hermitian:
        herm = 0.5 * (t_mat + t_mat.conj().T)
        evals, vecs = np.linalg.eigh(herm)
        return vecs @ (np.exp(exp_factor * time_difference * evals) * vecs[0].conj())
    return expm(exp_factor * time_difference * t_mat)[:, 0]


def krylov_exp_action(psi: np.ndarray,
                      action: Callable[[np.ndarray], np.ndarray],
                      time_difference: float,
                      exp_factor: complex,
                      krylov_tol: float = 1e-10,
                      max_krylov_dim: Union[int, None] = None,
                      hermitian: bool = True) -> np.ndarray:
    """
    Matrix-free Krylov (Arnoldi/Lanczos) exponential of an operator action.

    Returns ``exp(exp_factor * dt * A) @ psi`` where ``action(y)`` applies the operator
    ``A`` to a tensor ``y`` of ``psi``'s shape and ``exp_factor = -/+ i`` selects
    forward/backward evolution. This is the exact solution of the linear ODE
    ``y' = exp_factor * A y`` up to the Krylov truncation, so it replaces a many-substep
    RK solve of the same (constant-``A``) effective Schrödinger equation with a handful
    of action calls plus a tiny matrix exponential.

    Build. An Arnoldi recurrence with FULL re-orthogonalisation grows the orthonormal
    Krylov basis ``q_0 = psi/||psi||, q_1, ...`` of ``A`` seeded by ``psi``, recording
    the small projection ``T_m`` (upper-Hessenberg buffer). The propagated vector is
    ``||psi|| * sum_n [exp(exp_factor dt T_m) e_1]_n q_n``. The build itself is
    operator-agnostic (general upper-Hessenberg, not a Hermitian-specialised
    tridiagonal recurrence).

    The recurrence orthogonalises TWICE per step, accumulating both corrections into the
    same Hessenberg column. The second pass is for conditioning, not refinement: one pass
    of Gram-Schmidt loses orthogonality once the recurrence is driven deep on a
    rank-deficient seed, the projection ``T_m`` then no longer represents ``A`` on
    ``span(q)``, and its exponential is silently wrong -- the state stays normalised and
    nothing warns. Measured on a bond-1 product state at ``krylov_tol = 1e-14``, where a
    tight tolerance drives the depth: a single pass gave a global error of 1.3e-01
    against the dense propagator where two give 7.0e-09, at every bond dimension.

    Hermiticity. With ``hermitian=True`` (default) ``T_m`` is symmetrised and
    exponentiated by ``eigh`` in _krylov_small_propagator -- exact iff ``A`` is
    Hermitian, and the source of the norm-preserving real-time coefficients. With
    ``hermitian=False`` the general dense exponential of the raw ``T_m`` is used, for a
    genuinely non-Hermitian generator.

    Adaptive depth. The recurrence stops at the smallest depth ``m`` whose
    a-posteriori residual estimate ``||psi|| |h_{m+1,m}| |[exp(exp_factor dt T_m) e_1]_m|``
    falls below ``krylov_tol`` (the standard Saad bound), or at the Arnoldi breakdown
    (the Krylov subspace cannot exceed the ambient dimension, where the propagator is
    represented exactly) -- so it can never silently under-resolve. ``krylov_tol <= 0``
    disables the early stop, resolving exactly to breakdown. ``max_krylov_dim`` is an
    optional hard cap on ``m`` (default: the ambient dimension).
    """
    shape = psi.shape
    v0 = np.asarray(psi).reshape(-1)
    n = v0.size
    # A real seed under a real ``exp_factor`` (a real generator) keeps the whole
    # recurrence real; anything else promotes to complex.
    work_dtype = (np.result_type(v0.dtype, np.asarray(exp_factor).dtype, float)
                  if not np.iscomplexobj(v0) else complex)
    beta0 = float(np.linalg.norm(v0))
    if beta0 == 0.0:
        return np.array(psi, dtype=work_dtype, copy=True)
    max_steps = n if max_krylov_dim is None else min(n, int(max_krylov_dim))
    breakdown = np.finfo(float).eps * max(n, 1)
    breakdown_rel = 1e-12                              # happy-breakdown relative floor
    q = [v0 / beta0]                                   # q_0 (normalised seed)
    hess = np.zeros((4, 4), dtype=work_dtype)          # Hessenberg buffer, grows on demand
    m = 0                                              # completed Krylov dimension
    for j in range(max_steps):
        if j + 2 > hess.shape[0]:                      # double the buffer as depth grows
            grown = np.zeros((2 * hess.shape[0], 2 * hess.shape[0]), dtype=work_dtype)
            grown[:hess.shape[0], :hess.shape[0]] = hess
            hess = grown
        w = np.asarray(action(q[j].reshape(shape))).reshape(-1)
        w0_norm = float(np.linalg.norm(w))             # ||A q_j|| before re-orthogonalisation
        for i in range(j + 1):                         # full re-orthogonalisation
            hess[i, j] = np.vdot(q[i], w)
            w = w - hess[i, j] * q[i]
        for i in range(j + 1):                         # second pass: conditioning
            correction = np.vdot(q[i], w)
            hess[i, j] += correction
            w = w - correction * q[i]
        beta = float(np.linalg.norm(w))
        hess[j + 1, j] = beta
        m = j + 1
        # Happy breakdown: the Krylov subspace is (numerically) invariant, so the
        # propagator is already represented exactly on it. The threshold is relative
        # to ||A q_j||: continuing past it would form q_{j+1} = w / beta with beta at
        # the rounding floor of w, amplifying noise into a spurious direction. That
        # matters for rank-deficient seeds, where a tighter krylov_tol would otherwise
        # inject noise directions that downstream selection mistakes for populated ones.
        # An absolute floor guards the beta0-scaled case.
        if beta <= max(breakdown, breakdown_rel * w0_norm):
            break
        if krylov_tol > 0.0:                           # adaptive Lanczos-exp stop
            col = _krylov_small_propagator(hess[:m, :m], time_difference, exp_factor,
                                           hermitian=hermitian)
            if beta0 * beta * abs(col[m - 1]) < krylov_tol:
                break
        q.append(w / beta)
    coeffs = beta0 * _krylov_small_propagator(hess[:m, :m], time_difference, exp_factor,
                                              hermitian=hermitian)
    out = np.zeros(n, dtype=np.result_type(work_dtype, coeffs.dtype))
    for idx in range(m):
        out += coeffs[idx] * q[idx]
    return out.reshape(shape)
