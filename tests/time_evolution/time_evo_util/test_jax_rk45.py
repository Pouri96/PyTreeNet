"""
Tests for the JAX Dormand-Prince 5(4) propagator behind TimeEvoMode.JAX_RK45.

The propagator requires a GPU, which most machines running this suite do not
have. So that its mathematics is still covered, the tests are layered:

* the compiled kernels are device-agnostic and are driven directly;
* the public wrappers take an explicit ``device``, which the tests supply and
  ``TimeEvoMode`` never does, so the GPU-only contract holds in production
  while the wrappers are still exercised here;
* only assertions genuinely about GPU execution are skipped without one.
"""
import unittest
import subprocess
import sys
from unittest import mock

import numpy as np
from scipy.linalg import expm

from pytreenet.time_evolution.time_evolution import (EvoDirection, TimeEvoMode,
                                                     time_evolve)
from pytreenet.random import crandn
from pytreenet.random.random_matrices import random_hermitian_matrix

try:
    import jax
    import jax.numpy as jnp
    from pytreenet.time_evolution.time_evo_util import jax_rk45
    JAX_AVAILABLE = True
    GPU_AVAILABLE = jax_rk45.available_gpu() is not None
    # The wrappers pick a GPU when no device is named, which is what keeps the
    # mode GPU-only in production. Naming one is a supported argument, so the
    # tests name whatever is present.
    TEST_DEVICE = jax_rk45.available_gpu() or jax.devices()[0]
except ImportError:
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    TEST_DEVICE = None

needs_jax = unittest.skipUnless(JAX_AVAILABLE, "jax is not installed")
needs_gpu = unittest.skipUnless(JAX_AVAILABLE and GPU_AVAILABLE,
                                "no GPU visible to jax")

TOTAL_TIME = 1.0


def _problem(dimension: int = 8, seed: int = 1234):
    """
    Builds a reference problem with a known closed-form solution.

    The Hamiltonian is rescaled so its spectral norm times the evolution time
    is of order one. That keeps the coarse-step error clear of double precision
    noise while leaving the fine-step error above it, which is what makes the
    convergence order measurable.

    Returns:
        tuple: The Hermitian Hamiltonian, the generator -i*H, and the
            normalised initial state.
    """
    np.random.seed(seed)
    hamiltonian = random_hermitian_matrix(dimension)
    hamiltonian = hamiltonian * (2.0 / np.linalg.norm(hamiltonian, 2))
    psi = crandn(dimension)
    psi /= np.linalg.norm(psi)
    return hamiltonian, -1.0j * hamiltonian, psi


class TestStepPlan(unittest.TestCase):
    """
    Tests the splitting of an evolution time into equal steps.
    """

    @needs_jax
    def test_steps_multiply_back_to_the_evolution_time(self):
        # The requested final time must be reached exactly, not overshot.
        for num_step in range(1, 65):
            n_steps, dt = jax_rk45._step_plan(1.0, num_step)
            self.assertEqual(n_steps, num_step)
            self.assertAlmostEqual(n_steps * dt, 1.0, places=15)

    @needs_jax
    def test_zero_time_takes_no_steps(self):
        self.assertEqual(jax_rk45._step_plan(0.0, 4), (0, 0.0))

    @needs_jax
    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            jax_rk45._step_plan(-1.0, 4)
        with self.assertRaises(ValueError):
            jax_rk45._step_plan(1.0, 0)


class TestFixedStepAccuracy(unittest.TestCase):
    """
    Tests the fixed step policy against the matrix exponential.
    """

    def setUp(self):
        if not JAX_AVAILABLE:
            self.skipTest("jax is not installed")
        self.hamiltonian, self.generator, self.psi = _problem()

    def _run(self, num_step, time_difference=TOTAL_TIME, **kwargs):
        return jax_rk45.rk45_evolve_with_error(
            self.psi, self.generator, time_difference, num_step=num_step,
            device=TEST_DEVICE, **kwargs)

    def _exact(self, time_difference=TOTAL_TIME):
        return expm(self.generator * time_difference) @ self.psi

    def test_matches_matrix_exponential(self):
        state, _, _ = self._run(64)
        np.testing.assert_allclose(state, self._exact(), atol=1e-10)

    def test_fifth_order_convergence(self):
        step_counts = (2, 4, 8, 16)
        errors = [np.linalg.norm(self._run(n)[0] - self._exact())
                  for n in step_counts]
        slope = -np.polyfit(np.log2(step_counts), np.log2(errors), 1)[0]
        # A single-precision step size caps the attainable error near 1e-8 and
        # flattens this slope well below five.
        self.assertGreater(slope, 4.5, f"errors {errors}")
        self.assertLess(slope, 5.5, f"errors {errors}")

    def test_norm_is_preserved(self):
        state, _, _ = self._run(64)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=9)

    def test_error_estimate_is_fifth_order(self):
        # The embedded estimate must converge at the order of the method.
        # Approximating the seventh stage by the sixth, as a six-stage
        # implementation must, converges one order lower and gives ratios near
        # sixteen rather than thirty-two.
        estimates = [self._run(n)[1] for n in (8, 16, 32)]
        for coarse, fine in zip(estimates, estimates[1:]):
            self.assertGreater(coarse / fine, 20.0, f"estimates {estimates}")
            self.assertLess(coarse / fine, 48.0, f"estimates {estimates}")

    def test_error_estimate_brackets_the_true_error(self):
        state, max_abs, _ = self._run(5)
        true_error = np.linalg.norm(state - self._exact())
        self.assertGreater(max_abs, true_error / 10)
        self.assertLess(max_abs, true_error * 10)

    def test_scaled_error_respects_the_tolerance(self):
        # The scaled estimate is what the wrapper's warning keys off: at or
        # below one the step met the tolerance, above one it did not.
        _, _, met = self._run(64, rtol=1e-6, atol=1e-9)
        self.assertLess(met, 1.0)
        _, _, missed = self._run(2, rtol=1e-18, atol=1e-18)
        self.assertGreater(missed, 1.0)

    def test_negative_time_difference_evolves_backward(self):
        # This is how the TDVP algorithms express a backward half-step:
        # _update_site(node_id, -time_step_factor) rather than an EvoDirection.
        backward, _, _ = self._run(64, time_difference=-0.4)
        np.testing.assert_allclose(backward, self._exact(-0.4), atol=1e-10)

    def test_negative_time_difference_inverts_a_forward_step(self):
        forward, _, _ = self._run(64, time_difference=0.4)
        restored, _, _ = jax_rk45.rk45_evolve_with_error(
            forward, self.generator, -0.4, num_step=64, device=TEST_DEVICE)
        np.testing.assert_allclose(restored, self.psi, atol=1e-10)

    def test_zero_time_is_the_identity(self):
        state, max_abs, max_scaled = self._run(8, time_difference=0.0)
        np.testing.assert_allclose(state, self.psi, atol=0.0)
        self.assertEqual((max_abs, max_scaled), (0.0, 0.0))

    def test_state_shape_is_preserved(self):
        # The sweeps hand the propagator a site tensor, not a vector.
        tensor = self.psi.reshape(2, 2, 2)
        state, _, _ = jax_rk45.rk45_evolve_with_error(
            tensor, self.generator, TOTAL_TIME, num_step=64,
            device=TEST_DEVICE)
        self.assertEqual(state.shape, (2, 2, 2))
        np.testing.assert_allclose(state.reshape(-1), self._exact(),
                                   atol=1e-10)

    def test_warns_when_the_tolerance_is_missed(self):
        with self.assertWarns(RuntimeWarning):
            jax_rk45.rk45_evolve(self.psi, self.generator, TOTAL_TIME,
                                 num_step=2, rtol=1e-18, atol=1e-18,
                                 device=TEST_DEVICE)

    def test_rejects_an_unknown_option(self):
        # Explicit keyword parameters are what make a mistyped solver option a
        # loud error rather than a silently ignored one.
        with self.assertRaises(TypeError):
            jax_rk45.rk45_evolve(self.psi, self.generator, TOTAL_TIME,
                                 tol=1e-6, device=TEST_DEVICE)


class TestAdaptiveStepping(unittest.TestCase):
    """
    Tests the adaptive step size policy.
    """

    def setUp(self):
        if not JAX_AVAILABLE:
            self.skipTest("jax is not installed")
        self.hamiltonian, self.generator, self.psi = _problem()
        self.exact = expm(self.generator * TOTAL_TIME) @ self.psi

    def _run(self, rtol, atol, **kwargs):
        return jax_rk45.rk45_evolve_with_error(
            self.psi, self.generator, TOTAL_TIME, rtol=rtol, atol=atol,
            adaptive=True, device=TEST_DEVICE, **kwargs)

    def test_matches_matrix_exponential(self):
        state, _, _ = self._run(1e-12, 1e-14)
        np.testing.assert_allclose(state, self.exact, atol=1e-9)

    def test_tightening_the_tolerance_reduces_the_error(self):
        errors = []
        for rtol in (1e-4, 1e-6, 1e-8, 1e-10):
            state, _, _ = self._run(rtol, rtol * 1e-3)
            errors.append(np.linalg.norm(state - self.exact))
        for loose, tight in zip(errors, errors[1:]):
            self.assertLess(tight, loose, f"errors {errors}")

    def test_accepted_steps_meet_the_tolerance(self):
        _, _, max_scaled = self._run(1e-6, 1e-9)
        self.assertLessEqual(max_scaled, 1.0)

    def test_norm_is_preserved(self):
        state, _, _ = self._run(1e-10, 1e-13)
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=9)

    def test_evolves_backward_for_a_negative_duration(self):
        state, _, _ = jax_rk45.rk45_evolve_with_error(
            self.psi, self.generator, -0.4, rtol=1e-12, atol=1e-14,
            adaptive=True, device=TEST_DEVICE)
        np.testing.assert_allclose(state, expm(self.generator * -0.4) @ self.psi,
                                   atol=1e-9)

    def test_warns_when_the_step_budget_is_exhausted(self):
        # An unreachable tolerance must stop and say so, not loop forever.
        with self.assertWarns(RuntimeWarning):
            jax_rk45.rk45_evolve(self.psi, self.generator, TOTAL_TIME,
                                 rtol=1e-14, atol=1e-18, adaptive=True,
                                 max_steps=5, device=TEST_DEVICE)

    def test_initial_step_estimate_is_sane(self):
        estimate = jax_rk45._estimate_initial_step_size(
            self.generator, self.psi, TOTAL_TIME, 1e-6, 1e-9)
        self.assertGreater(estimate, 0.0)
        self.assertLessEqual(estimate, 0.1 * TOTAL_TIME)


class TestCompilationCaching(unittest.TestCase):
    """
    Tests that the integration loop compiles once per state dimension.

    This is the optimisation the mode exists for. A TDVP sweep calls the
    propagator once per site, with a step count that may vary, so the step
    count, step size and tolerances are traced values inside the kernel and
    must not trigger a recompilation.
    """

    def setUp(self):
        if not JAX_AVAILABLE:
            self.skipTest("jax is not installed")
        self.hamiltonian, self.generator, self.psi = _problem()

    def _call(self, generator, psi, num_step, time_difference=1.0, rtol=1e-10):
        return jax_rk45.rk45_evolve_with_error(
            psi, generator, time_difference, num_step=num_step, rtol=rtol,
            device=TEST_DEVICE)

    def test_step_count_and_tolerances_do_not_recompile(self):
        kernel = jax_rk45._get_fixed_kernel()
        if not hasattr(kernel, '_cache_size'):
            self.skipTest("this jax version exposes no jit cache size")

        self._call(self.generator, self.psi, 8)
        # Relative to a baseline rather than equal to one: the kernel is
        # memoised in a module global that earlier tests have populated.
        baseline = kernel._cache_size()

        self._call(self.generator, self.psi, 37)
        self._call(self.generator, self.psi, 1000)
        self.assertEqual(kernel._cache_size(), baseline,
                         "changing the step count recompiled the kernel")

        self._call(self.generator, self.psi, 8, time_difference=0.5)
        self.assertEqual(kernel._cache_size(), baseline,
                         "changing the step size recompiled the kernel")

        self._call(self.generator, self.psi, 8, rtol=1e-6)
        self.assertEqual(kernel._cache_size(), baseline,
                         "changing the tolerance recompiled the kernel")

        _, other_generator, other_psi = _problem(dimension=16, seed=99)
        self._call(other_generator, other_psi, 8)
        self.assertEqual(kernel._cache_size(), baseline + 1,
                         "a new state dimension should compile exactly once")


class TestModeSurface(unittest.TestCase):
    """
    Tests how the mode presents itself through TimeEvoMode.

    None of these need jax: they are pure enum logic.
    """

    def test_is_neither_scipy_nor_action_evolvable(self):
        # Staying out of action_evolvable is what makes
        # effective_single_site_evolution assemble the dense effective
        # Hamiltonian, which is what the GPU kernel needs.
        self.assertFalse(TimeEvoMode.JAX_RK45.is_scipy())
        self.assertFalse(TimeEvoMode.JAX_RK45.action_evolvable())
        self.assertTrue(TimeEvoMode.JAX_RK45.is_jax())

    def test_no_other_mode_is_jax(self):
        for mode in TimeEvoMode:
            if mode is not TimeEvoMode.JAX_RK45:
                self.assertFalse(mode.is_jax(), f"{mode.name} reports is_jax")

    def test_is_not_the_default_and_not_the_fastest(self):
        self.assertIs(TimeEvoMode.fastest_equivalent(), TimeEvoMode.CHEBYSHEV)
        self.assertFalse(TimeEvoMode.FASTEST.is_jax())

    def test_value_collides_with_no_other_backend(self):
        # The value is handed verbatim to solve_ivp's method and to
        # fast_exp_action's mode, so it must belong to neither.
        self.assertEqual(TimeEvoMode.JAX_RK45.value, "jax_rk45")
        others = {m.value for m in TimeEvoMode if m is not TimeEvoMode.JAX_RK45}
        self.assertNotIn(TimeEvoMode.JAX_RK45.value, others)

    def test_declares_its_solver_options(self):
        defaults = TimeEvoMode.default_solver_options(TimeEvoMode.JAX_RK45)
        self.assertEqual(defaults, {'atol': 1e-10, 'rtol': 1e-10,
                                    'num_step': 1, 'adaptive': False})
        valid = TimeEvoMode.valid_solver_options(TimeEvoMode.JAX_RK45)
        self.assertLessEqual(set(defaults), valid)
        self.assertIn('max_steps', valid)
        self.assertIn('exp_factor', valid)


class TestModeDispatch(unittest.TestCase):
    """
    Tests that time_evolve hands the propagator the right arguments.

    Needs no device: the propagator is replaced by a recorder. This pins down
    that the mode's options and the evolution direction actually reach the
    solver.
    """

    def setUp(self):
        if not JAX_AVAILABLE:
            self.skipTest("jax is not installed")
        self.hamiltonian, self.generator, self.psi = _problem()

    def _dispatch(self, caller, **kwargs):
        with mock.patch.object(jax_rk45, 'rk45_evolve',
                               return_value=self.psi) as recorder:
            caller(**kwargs)
        self.assertEqual(recorder.call_count, 1)
        return recorder.call_args

    def test_module_time_evolve_applies_the_mode_defaults(self):
        args, kwargs = self._dispatch(
            lambda: time_evolve(self.psi, self.hamiltonian, 0.4,
                                mode=TimeEvoMode.JAX_RK45))
        np.testing.assert_allclose(args[1], self.generator)
        self.assertEqual(args[2], 0.4)
        self.assertEqual(kwargs, {'atol': 1e-10, 'rtol': 1e-10,
                                  'num_step': 1, 'adaptive': False})

    def test_module_time_evolve_honours_explicit_options(self):
        _, kwargs = self._dispatch(
            lambda: time_evolve(self.psi, self.hamiltonian, 0.4,
                                mode=TimeEvoMode.JAX_RK45,
                                num_step=32, adaptive=True, rtol=1e-7))
        self.assertEqual(kwargs['num_step'], 32)
        self.assertTrue(kwargs['adaptive'])
        self.assertEqual(kwargs['rtol'], 1e-7)
        self.assertEqual(kwargs['atol'], 1e-10)

    def test_backward_direction_flips_the_generator(self):
        args, _ = self._dispatch(
            lambda: time_evolve(self.psi, self.hamiltonian, 0.4,
                                forward=EvoDirection.BACKWARD,
                                mode=TimeEvoMode.JAX_RK45))
        np.testing.assert_allclose(args[1], -self.generator)

    def test_exp_factor_override_reaches_the_generator(self):
        # exp_factor is consumed by the dispatch, not forwarded as an option.
        args, kwargs = self._dispatch(
            lambda: TimeEvoMode.JAX_RK45.time_evolve(
                self.psi, self.hamiltonian, 0.4, exp_factor=2.0j))
        np.testing.assert_allclose(args[1], 2.0j * self.hamiltonian)
        self.assertNotIn('exp_factor', kwargs)

    def test_enum_method_passes_no_defaults_of_its_own(self):
        # Only the module-level time_evolve resolves default_solver_options, so
        # the TDVP path arrives with nothing and relies on the propagator's own
        # signature defaults.
        _, kwargs = self._dispatch(
            lambda: TimeEvoMode.JAX_RK45.time_evolve(
                self.psi, self.hamiltonian, 0.4))
        self.assertEqual(kwargs, {})


class TestOptionalDependency(unittest.TestCase):
    """
    Tests that jax stays an optional dependency.
    """

    def test_importing_pytreenet_does_not_import_jax(self):
        # Deliberately not guarded on JAX_AVAILABLE: this must hold whether or
        # not jax is installed, since importing it costs seconds and it is not
        # a declared runtime dependency.
        script = ("import sys, pytreenet\n"
                  "assert 'jax' not in sys.modules, 'pytreenet imported jax'\n")
        subprocess.run([sys.executable, '-c', script], check=True)

    def test_other_modes_do_not_need_jax(self):
        hamiltonian, generator, psi = _problem()
        exact = expm(generator * 0.4) @ psi
        for name in ('EXPM', 'CHEBYSHEV', 'RK45', 'DOP853', 'KRYLOV'):
            state = time_evolve(psi, hamiltonian, 0.4,
                                mode=getattr(TimeEvoMode, name))
            np.testing.assert_allclose(state, exact, atol=1e-8,
                                       err_msg=f"mode {name}")


class TestGpuRequirement(unittest.TestCase):
    """
    Tests that the mode insists on a GPU when no device is named.
    """

    @unittest.skipUnless(JAX_AVAILABLE, "jax is not installed")
    @unittest.skipIf(GPU_AVAILABLE, "a GPU is visible, so nothing to refuse")
    def test_propagator_refuses_without_a_gpu(self):
        _, generator, psi = _problem()
        with self.assertRaises(RuntimeError) as caught:
            jax_rk45.rk45_evolve(psi, generator, TOTAL_TIME, num_step=8)
        message = str(caught.exception)
        self.assertIn('GPU', message)
        # The refusal has to say what to do instead.
        self.assertIn('RK45', message)
        self.assertIn('KRYLOV', message)

    @unittest.skipUnless(JAX_AVAILABLE, "jax is not installed")
    @unittest.skipIf(GPU_AVAILABLE, "a GPU is visible, so nothing to refuse")
    def test_time_evolve_refuses_the_jax_mode_without_a_gpu(self):
        hamiltonian, _, psi = _problem()
        with self.assertRaises(RuntimeError):
            time_evolve(psi, hamiltonian, 0.4, mode=TimeEvoMode.JAX_RK45)


class TestGpuExecution(unittest.TestCase):
    """
    Tests that are genuinely about running on a GPU. Requires one.
    """

    def setUp(self):
        if not (JAX_AVAILABLE and GPU_AVAILABLE):
            self.skipTest("no GPU visible to jax")
        self.hamiltonian, self.generator, self.psi = _problem()

    @needs_gpu
    def test_selected_device_is_a_gpu(self):
        self.assertIn(jax_rk45.require_gpu_device().platform,
                      ('gpu', 'cuda', 'rocm'))

    @needs_gpu
    def test_agrees_with_the_cpu_modes(self):
        exact = expm(self.generator * 0.4) @ self.psi
        fixed = time_evolve(self.psi, self.hamiltonian, 0.4,
                            mode=TimeEvoMode.JAX_RK45, num_step=32)
        np.testing.assert_allclose(fixed, exact, atol=1e-8)
        adaptive = time_evolve(self.psi, self.hamiltonian, 0.4,
                               mode=TimeEvoMode.JAX_RK45, adaptive=True,
                               rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(adaptive, exact, atol=1e-8)

    @needs_gpu
    def test_backward_evolution_inverts_forward_evolution(self):
        forward = time_evolve(self.psi, self.hamiltonian, 0.4,
                              mode=TimeEvoMode.JAX_RK45, num_step=64)
        restored = time_evolve(forward, self.hamiltonian, 0.4,
                               forward=EvoDirection.BACKWARD,
                               mode=TimeEvoMode.JAX_RK45, num_step=64)
        np.testing.assert_allclose(restored, self.psi, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
