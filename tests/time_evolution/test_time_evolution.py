import unittest

import numpy as np
from scipy.linalg import expm
import pytest

import pytreenet as ptn
from pytreenet.time_evolution.time_evolution import (EvoDirection, TimeEvoMode,
                                                     time_evolve)
from pytreenet.time_evolution.results import Results
from pytreenet.random import crandn
from pytreenet.random.random_matrices import random_hermitian_matrix

class TestTimeEvolutionInit(unittest.TestCase):

    def setUp(self):
        # Initialise initial state
        self.initial_state = ptn.TreeTensorNetworkState()
        self.initial_state.add_root(ptn.Node(identifier="root"), crandn((5,6,2)))
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c1"),
            crandn((5,3)), 0, "root", 0)
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c2"),
            crandn((6,4)), 0, "root", 1)

        # Operators
        single_site_operator = ptn.TensorProduct({"root": crandn((2,2))})
        two_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                               "c2": crandn((4,4))})
        three_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                                  "c2": crandn((4,4)),
                                                  "root": crandn((2,2))})
        self.operators = [single_site_operator,
                          two_site_operator,
                          three_site_operator]

        # Time Parameters
        self.time_step_size = 0.1
        self.final_time = 1.0

        self.time_evol = ptn.TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators)

    def test_initial_state_init(self):
        for identifier in ["root", "c1", "c2"]:
            self.assertTrue(identifier in self.time_evol.initial_state.nodes)

    def test_time_init(self):
        self.assertEqual(self.time_step_size, self.time_evol.time_step_size)
        self.assertEqual(self.final_time, self.time_evol.final_time)
        self.assertEqual(10, self.time_evol.num_time_steps)

    def test_compute_num_time_steps(self):
        num_time_steps = self.time_evol._compute_num_time_steps()
        self.assertEqual(10, num_time_steps)

    def test_operator_init(self):
        self.assertEqual(len(self.operators), len(self.time_evol.operators))
        for i, ten_prod in enumerate(self.operators):
            self.assertIn(str(i), self.time_evol.operators)
            self.assertEqual(ten_prod, self.time_evol.operators[str(i)])

    def test_time_step_size_check(self):
        self.assertRaises(ValueError, ptn.TimeEvolution, self.initial_state,
            -0.1, self.final_time, self.operators)

    def test_final_time_check(self):
        self.assertRaises(ValueError, ptn.TimeEvolution, self.initial_state,
            self.time_step_size, -1, self.operators)

    def test_only_one_operator(self):
        time_evo = ptn.TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators[0])
        self.assertTrue(isinstance(time_evo.operators, dict))
        self.assertEqual(len(time_evo.operators), 1)
        self.assertEqual(time_evo.operators["0"], self.operators[0])

    def test_results(self):
        '''
        The results are only initialised once a time evolution is run.
        '''
        self.time_evol.results.close_to(Results())

class TestTimeEvolutionMethods(unittest.TestCase):
    def setUp(self):
        # Initialise initial state
        self.initial_state = ptn.TreeTensorNetworkState()
        self.initial_state.add_root(ptn.Node(identifier="root"), crandn((5,6,2)))
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c1"),
            crandn((5,3)), 0, "root", 0)
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c2"),
            crandn((6,4)), 0, "root", 1)

        # Operators
        single_site_operator = ptn.TensorProduct({"root": crandn((2,2))})
        two_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                               "c2": crandn((4,4))})
        three_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                                  "c2": crandn((4,4)),
                                                  "root": crandn((2,2))})
        self.operators = [single_site_operator,
                          two_site_operator,
                          three_site_operator]

        # Time Parameters
        self.time_step_size = 0.1
        self.final_time = 1.0

        self.time_evol = ptn.TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators)

    def test_run_one_time_step(self):
        self.assertRaises(NotImplementedError, self.time_evol.run_one_time_step)

    def test_init_results(self):
        """
        Initialising the results with standard evaluation_time.
        """
        self.time_evol.init_results()
        results = self.time_evol.results
        self.assertTrue(results.is_initialized())
        self.assertEqual(len(self.time_evol.operators) + 1, results.num_results())
        self.assertEqual(self.time_evol.num_time_steps + 1, results.results_length())

    def test_init_results_with_custom_evaluation_time(self):
        """
        Initialising the results with custom evaluation_time.
        """
        evaluation_time = 2
        self.time_evol.init_results(evaluation_time=evaluation_time)
        results = self.time_evol.results
        self.assertTrue(results.is_initialized())
        self.assertEqual(len(self.time_evol.operators) + 1, results.num_results())
        self.assertEqual(self.time_evol.num_time_steps // 2 + 1, results.results_length())

    def test_init_results_with_inf(self):
        """
        Initialising the results with evalutation_time = "inf".
        Thus only the last result is saved.
        """
        evaluation_time = "inf"
        self.time_evol.init_results(evaluation_time=evaluation_time)
        results = self.time_evol.results
        self.assertTrue(results.is_initialized())
        self.assertEqual(len(self.time_evol.operators) + 1, results.num_results())
        self.assertEqual(2, results.results_length())


### Test EvoDirection Enum Class
@pytest.mark.parametrize("test_input,expected", [(True, EvoDirection.FORWARD),
                                                (False, EvoDirection.BACKWARD)])
def test_from_bool(test_input, expected):
    """
    Test the from_bool method of the EvoDirection enum class.
    """
    assert EvoDirection.from_bool(test_input) == expected

@pytest.mark.parametrize("test_input,expected", [(EvoDirection.FORWARD, -1),
                                                (EvoDirection.BACKWARD, 1)])
def test_exp_sign(test_input, expected):
    """
    Test the exp_sign method of the EvoDirection enum class.
    """
    assert EvoDirection.exp_sign(test_input) == expected

@pytest.mark.parametrize("test_input,expected", [(EvoDirection.FORWARD, -1j),
                                                (EvoDirection.BACKWARD, 1j)])
def test_exp_factor(test_input, expected):
    """
    Test the exp_factor method of the EvoDirection enum class.
    """
    assert EvoDirection.exp_factor(test_input) == expected


### Test TimeEvoMode Enum Class
def test_fastest_equivalent():
    """
    Test that the 'fastest' mode is equivalent to 'chebyshev'.
    """
    assert TimeEvoMode.fastest_equivalent() == TimeEvoMode.CHEBYSHEV

@pytest.mark.parametrize("mode,expected", [
    (TimeEvoMode.EXPM, False),
    (TimeEvoMode.CHEBYSHEV, False),
    (TimeEvoMode.SPARSE, False),
    (TimeEvoMode.RK45, True),
    (TimeEvoMode.RK23, True),
    (TimeEvoMode.DOP853, True),
    (TimeEvoMode.BDF, True),
])
def test_is_scipy(mode, expected):
    """
    Test the is_scipy method of the TimeEvoMode enum class.
    """
    assert mode.is_scipy() == expected

@pytest.mark.parametrize("mode", [
    TimeEvoMode.RK45,
    TimeEvoMode.RK23,
    TimeEvoMode.DOP853,
    TimeEvoMode.BDF,
])
def test_is_scipy_raises(mode):
    """
    Test that is_scipy raises an error for non-scipy modes.
    """
    shape = (2, 3, 4)
    psi_init = crandn(shape)
    dt = 0.1
    exponent_mat = random_hermitian_matrix(np.prod(shape).item())
    exponent_tens = exponent_mat.reshape(2*list(shape))
    def multiply_fn(_, x):
        return np.tensordot(exponent_tens, x, axes=([3,4,5], [0,1,2]))
    found = mode.time_evolve_action(psi_init, multiply_fn, dt,
                                    atol = 1e-9, rtol = 1e-9)
    # Referemce is the exponentiation of the matrix
    reference = expm(-1j * exponent_mat * dt) @ psi_init.flatten()
    # Check if the results are close
    assert shape == found.shape
    np.testing.assert_allclose(found.flatten(), reference)

@pytest.mark.parametrize("mode", [
    TimeEvoMode.EXPM,
    TimeEvoMode.CHEBYSHEV,
    TimeEvoMode.SPARSE,
    TimeEvoMode.RK45,
    TimeEvoMode.RK23,
    TimeEvoMode.DOP853,
    TimeEvoMode.BDF
])
def test_time_evolve(mode):
    """
    Test the time_evolve method of the TimeEvoMode enum class that simply uses
    the full Hamiltonian matrix.
    """
    shape = (2, 3, 4)
    psi_init = crandn(shape)
    dt = 0.1
    exponent_mat = random_hermitian_matrix(np.prod(shape).item())
    found = mode.time_evolve(psi_init, exponent_mat, dt,
                             atol = 1e-9, rtol = 1e-9)
    # Referemce is the exponentiation of the matrix
    reference = expm(-1j * exponent_mat * dt) @ psi_init.flatten()
    # Check if the results are close
    assert shape == found.shape
    np.testing.assert_allclose(found.flatten(), reference)

@pytest.mark.parametrize("mode", [
    TimeEvoMode.RK45,
    TimeEvoMode.RK23,
    TimeEvoMode.DOP853,
    TimeEvoMode.BDF
])
def test_module_time_evolve_applies_the_mode_defaults(mode):
    """
    Test that the module-level time_evolve resolves the mode's own solver options
    when the caller supplies none.

    Without them a scipy mode runs at solve_ivp's default rtol=1e-3, which returns
    a plausible trajectory rather than an error, so only the accuracy shows it.

    The step is a full unit of time because a short one hides the difference: at
    dt=0.1 the higher-order pairs are already near 1e-7 even at rtol=1e-3, so the
    assertion would not separate the two. At dt=1.0 every scipy mode lands at 5e-4
    or worse on the defaults and at 2e-8 or better on the mode's own options.
    """
    shape = (2, 3, 4)
    psi_init = crandn(shape)
    dt = 1.0
    exponent_mat = random_hermitian_matrix(np.prod(shape).item())
    found = time_evolve(psi_init, exponent_mat, dt, mode=mode)
    reference = expm(-1j * exponent_mat * dt) @ psi_init.flatten()
    assert shape == found.shape
    np.testing.assert_allclose(found.flatten(), reference, atol=1e-6)

@pytest.mark.parametrize("mode", [TimeEvoMode.RK45, TimeEvoMode.KRYLOV])
def test_module_time_evolve_honours_explicit_options(mode):
    """
    Test that options passed to the module-level time_evolve reach the solver,
    overlaid on the mode's defaults rather than replacing them.
    """
    shape = (2, 3, 4)
    psi_init = crandn(shape)
    dt = 0.1
    exponent_mat = random_hermitian_matrix(np.prod(shape).item())
    opts = ({"krylov_tol": 1e-12} if mode is TimeEvoMode.KRYLOV
            else {"atol": 1e-12})
    found = time_evolve(psi_init, exponent_mat, dt, mode=mode, **opts)
    reference = expm(-1j * exponent_mat * dt) @ psi_init.flatten()
    assert shape == found.shape
    np.testing.assert_allclose(found.flatten(), reference, atol=1e-8)

if __name__ == "__main__":
    unittest.main()
