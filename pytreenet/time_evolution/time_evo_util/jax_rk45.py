"""
A GPU propagator for the effective Schroedinger equation, using JAX.

This module backs :class:`TimeEvoMode.JAX_RK45`. It solves the linear initial
value problem

    dy/dt = G y,    y(0) = psi

over one time step, where ``G`` is the generator ``exp_factor * hamiltonian``
assembled by the caller. The integrator is Dormand-Prince 5(4), the same method
as scipy's ``RK45``, with two stepping policies behind one entry point:

* fixed step, ``num_step`` steps of equal size, and
* adaptive step, with the step size driven by the embedded error estimate
  against ``rtol`` and ``atol``.

Why this mode forms the operator
--------------------------------
The matrix-free modes (``KRYLOV`` and the scipy family) consume a contraction
closure and never build the effective Hamiltonian. This one deliberately does
the opposite: it takes the assembled matrix, ships it to the GPU once, and runs
every stage of every step on the device. Driving it from a NumPy closure instead
would copy the state back to the host for each of the six matrix-vector products
per step, which costs more than it saves and makes the single-compilation
optimisation below impossible. So ``JAX_RK45`` is not ``action_evolvable``, and
:func:`effective_single_site_evolution` assembles the matrix for it.

The method is FSAL, "first same as last": its seventh stage row equals its fifth
order weights, so stage seven is exactly ``f(t + h, y_new)``, which is also the
next step's first stage. Carrying that vector through the loop buys an exact
embedded error estimate for six matrix-vector products per step -- the same count
a naive six-stage implementation pays for an estimate that is one order too low.

Performance
-----------
The whole integration loop lives inside a single ``jax.jit``-compiled kernel, and
the step count, step size and tolerances are all passed as *traced* values. The
kernel therefore compiles once per state dimension and is reused for every step
count, which matters because a TDVP sweep calls this once per site per sweep.
:func:`configure_jax` additionally enables JAX's persistent on-disk compilation
cache, so even the first compilation is paid once per machine rather than once
per process.

This module requires a GPU. Without one, use ``TimeEvoMode.RK45`` or
``TimeEvoMode.DOP853`` for the same Runge-Kutta families on the CPU, or
``TimeEvoMode.KRYLOV`` for a matrix-free propagator.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple, Union
from warnings import warn

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ['rk45_evolve', 'rk45_evolve_with_error', 'configure_jax',
           'available_gpu', 'require_gpu_device']

# --------------------------------------------------------------------------- #
# Dormand-Prince 5(4) tableau
# --------------------------------------------------------------------------- #
# Stage coefficients. The implied time nodes are c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1].
_A21 = 1/5
_A31, _A32 = 3/40, 9/40
_A41, _A42, _A43 = 44/45, -56/15, 32/9
_A51, _A52, _A53, _A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
_A61, _A62, _A63, _A64, _A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

# Fifth order weights. b2 and b7 vanish and are dropped rather than multiplied
# in. This row is also the seventh stage row a7j, which is what makes the method
# FSAL.
_B1, _B3, _B4, _B5, _B6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

# Embedded error weights, b(5th order) - b(4th order), using the fourth order
# companion [1951/21600, 0, 22642/50085, 451/720, -12231/42400, 649/6300, 1/60],
# as diffrax's dopri5 does. e2 vanishes and is dropped.
#
# Note when comparing error magnitudes against TimeEvoMode.RK45: scipy uses a
# different, equally valid fourth order companion, and this row is exactly -2/3
# times scipy's. The two estimates differ by that constant factor while
# converging at the same rate.
_E1 = 35/384 - 1951/21600
_E3 = 500/1113 - 22642/50085
_E4 = 125/192 - 451/720
_E5 = -2187/6784 + 12231/42400
_E6 = 11/84 - 649/6300
_E7 = -1/60

# Order of the error estimate, setting the adaptive step size exponent.
_ERROR_ORDER = 5

# Adaptive controller constants, following Hairer, Norsett & Wanner.
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 5.0

_GPU_PLATFORMS = ('cuda', 'rocm', 'gpu')
_DEFAULT_CACHE_DIR = '/tmp/jax_cache'

_CONFIGURED = False
_COMPILED_FIXED_KERNEL = None
_COMPILED_ADAPTIVE_KERNEL = None


# --------------------------------------------------------------------------- #
# Configuration and device selection
# --------------------------------------------------------------------------- #
def available_gpu():
    """
    Finds a GPU device for JAX, if there is one.

    Returns:
        The first GPU-class device JAX can see, or None if it can see none.
            Never raises, so it is safe to use as a probe.
    """
    for platform in _GPU_PLATFORMS:
        try:
            devices = jax.devices(platform)
        except RuntimeError:
            # JAX raises rather than returning an empty list when a backend is
            # not built into the installed wheel at all.
            continue
        if devices:
            return devices[0]
    return None


def require_gpu_device():
    """
    Finds the GPU device to run on, insisting that one exists.

    Returns:
        The first GPU-class device JAX can see.

    Raises:
        RuntimeError: If JAX cannot see a GPU, naming the devices it can see
            and the modes that do not need one.
    """
    device = available_gpu()
    if device is None:
        raise RuntimeError(
            'TimeEvoMode.JAX_RK45 requires a GPU, but JAX cannot see one. The '
            f'default backend is {jax.default_backend()!r} and the visible '
            f'devices are {jax.devices()}. Either install a GPU build of JAX\n'
            "    pip install 'pytreenet[jax-cuda]'\n"
            'see https://docs.jax.dev/en/latest/installation.html, or use a '
            'mode that runs on the CPU: TimeEvoMode.RK45 and '
            'TimeEvoMode.DOP853 are the same Runge-Kutta families via scipy, '
            'and TimeEvoMode.KRYLOV is matrix-free.')
    return device


def configure_jax(cache_dir: Union[str, None] = None) -> None:
    """
    Configures JAX for the RK45 propagator.

    Enables 64 bit precision and high precision matrix products, and turns on
    the persistent on-disk compilation cache so the kernels are compiled once
    per machine rather than once per process. Idempotent, so callers need not
    coordinate; the propagator calls it itself on first use.

    See https://docs.jax.dev/en/latest/persistent_compilation_cache.html

    Args:
        cache_dir (str, optional): Directory for the persistent compilation
            cache. Defaults to the ``PYTREENET_JAX_CACHE_DIR`` environment
            variable, or ``/tmp/jax_cache`` if that is unset.

    Note:
        This deliberately never sets ``jax_platforms``. That option is
        process-global, and pinning it to ``'gpu'`` makes every later backend
        access raise on a host where the GPU backend is not built in, which
        breaks unrelated JAX code sharing the interpreter. The device is chosen
        per call instead, by :func:`require_gpu_device`.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Load-bearing: with x64 disabled, jnp.asarray of a complex128 array
    # silently yields complex64, costing eight digits and making the embedded
    # error estimate meaningless.
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'high')

    cache_dir = (cache_dir
                 or os.environ.get('PYTREENET_JAX_CACHE_DIR')
                 or _DEFAULT_CACHE_DIR)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', str(cache_dir))
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    # jax_persistent_cache_enable_xla_caches already defaults to
    # "xla_gpu_per_fusion_autotune_cache_dir", so it is left alone.

    if available_gpu() is not None:
        flags = os.environ.get('XLA_FLAGS', '')
        if '--xla_gpu_use_cudnn_batchnorm' not in flags:
            os.environ['XLA_FLAGS'] = \
                f'{flags} --xla_gpu_use_cudnn_batchnorm=true'.strip()

    _CONFIGURED = True


def _require_x64() -> None:
    """
    Checks that JAX is still in 64 bit mode.

    Raises:
        RuntimeError: If 64 bit mode was disabled after configuration.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            'jax_enable_x64 is disabled, so a complex128 state would silently '
            'be downcast to complex64 and the RK45 error estimate would be '
            'meaningless. Do not disable it after pytreenet has configured '
            'JAX.')


# --------------------------------------------------------------------------- #
# Step planning
# --------------------------------------------------------------------------- #
def _step_plan(time_difference: float, num_step: int) -> Tuple[int, float]:
    """
    Splits an evolution time into equally sized steps.

    Args:
        time_difference (float): The total evolution time. Must not be
            negative; a backward evolution is folded into the sign of the
            generator by the caller.
        num_step (int): The number of steps to take.

    Returns:
        Tuple[int, float]: The step count and the step size. Their product is
            ``time_difference`` by construction, so the requested final time is
            reached exactly rather than overshot.

    Raises:
        ValueError: If ``time_difference`` is negative or ``num_step`` is below
            one.
    """
    if time_difference < 0.0:
        raise ValueError('time_difference must not be negative, got '
                         f'{time_difference!r}; fold the sign into the '
                         'generator instead')
    if time_difference == 0.0:
        return 0, 0.0
    n_steps = int(num_step)
    if n_steps < 1:
        raise ValueError(f'num_step must be at least 1, got {num_step!r}')
    return n_steps, time_difference / n_steps


def _estimate_initial_step_size(generator: np.ndarray, y0: np.ndarray,
                                t_span: float, rtol: float, atol: float,
                                p: int = _ERROR_ORDER) -> float:
    """
    Estimates a starting step size for the adaptive policy.

    Implements the algorithm of Section II.4 of Hairer, Norsett & Wanner,
    "Solving Ordinary Differential Equations I: Nonstiff Problems" (2nd
    edition, 1993), as used by scipy's RK45 and MATLAB's ode45.

    Uses NumPy rather than JAX: it runs once per integration, and inside the
    compiled region it would force a host-device round trip.

    Args:
        generator (np.ndarray): The generator as a matrix.
        y0 (np.ndarray): The initial state as a vector.
        t_span (float): The total integration time.
        rtol (float): The relative tolerance.
        atol (float): The absolute tolerance.
        p (int, optional): The order of the method. Defaults to 5.

    Returns:
        float: The estimated initial step size.
    """
    gen = np.asarray(generator)
    y = np.asarray(y0)

    f0 = gen @ y
    scale = atol + rtol * np.abs(y)

    d0 = np.sqrt(np.mean(np.abs(y / scale) ** 2))
    d1 = np.sqrt(np.mean(np.abs(f0 / scale) ** 2))

    if (d0 < 1e-5) or (d1 < 1e-5):
        h0 = 1e-6
    else:
        h0 = 0.01 * (d0 / d1)

    # One explicit Euler step, to finite-difference the second derivative.
    y1 = y + h0 * f0
    f1 = gen @ y1
    d2 = np.sqrt(np.mean(np.abs((f1 - f0) / (h0 * scale)) ** 2))

    max_d1_d2 = max(d1, d2)
    if max_d1_d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max_d1_d2) ** (1.0 / (p + 1))

    h = min(100 * h0, h1)
    # Never start with more than a tenth of the interval, nor so small that the
    # step count explodes.
    h = min(h, 0.1 * t_span)
    return float(max(h, 1e-12))


# --------------------------------------------------------------------------- #
# The step
# --------------------------------------------------------------------------- #
def _dp5_step(gen, y, k1, dt):
    """
    Takes one Dormand-Prince 5(4) step.

    Args:
        gen: The generator as a matrix.
        y: The current state as a vector.
        k1: The first stage, ``gen @ y``. Passed in rather than recomputed,
            because the previous step already produced it.
        dt: The step size.

    Returns:
        tuple: The fifth order solution, the seventh stage, and the embedded
            error estimate. The seventh stage is the next step's ``k1``.
    """
    k2 = gen @ (y + dt * _A21 * k1)
    k3 = gen @ (y + dt * (_A31 * k1 + _A32 * k2))
    k4 = gen @ (y + dt * (_A41 * k1 + _A42 * k2 + _A43 * k3))
    k5 = gen @ (y + dt * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4))
    k6 = gen @ (y + dt * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4
                          + _A65 * k5))
    y_new = y + dt * (_B1 * k1 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6)
    # FSAL. The seventh node is c7 = 1 and the seventh stage row equals the
    # fifth order weights, so this is exactly f(t + dt, y_new) with no
    # approximation, and exactly the next step's first stage.
    k7 = gen @ y_new
    error = dt * (_E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6
                  + _E7 * k7)
    return y_new, k7, error


def _scaled_error_norm(error, y, y_new, rtol, atol, inv_sqrt_n):
    """
    Reduces an error estimate to a dimensionless number against a tolerance.

    Uses the mixed absolute/relative RMS convention of ``scipy.integrate``, so
    a value at or below one means the step met the requested tolerance.
    """
    scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
    scale = jnp.maximum(scale, jnp.finfo(jnp.float64).tiny)
    return (jnp.linalg.norm(error / scale) * inv_sqrt_n).astype(jnp.float64)


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def _fixed_step_kernel(generator, psi, dt, n_steps, rtol, atol):
    """
    Integrates with a fixed step size.

    Every argument is traced, so changing the step count, step size or
    tolerances reuses the compiled kernel instead of recompiling it.

    Args:
        generator: The generator as a matrix.
        psi: The initial state as a vector.
        dt: The step size.
        n_steps: The number of steps.
        rtol: The relative tolerance, used only to scale the reported error.
        atol: The absolute tolerance, used only to scale the reported error.

    Returns:
        tuple: The final state, the largest per-step error norm, and the
            largest per-step scaled error norm.
    """
    gen = jnp.asarray(generator, dtype=jnp.complex128)
    y0 = jnp.asarray(psi, dtype=jnp.complex128)
    inv_sqrt_n = 1.0 / jnp.sqrt(jnp.asarray(y0.size, dtype=jnp.float64))

    def body(_, carry):
        y, k1, max_abs, max_scaled = carry
        y_new, k7, error = _dp5_step(gen, y, k1, dt)
        abs_norm = jnp.linalg.norm(error).astype(jnp.float64)
        scaled = _scaled_error_norm(error, y, y_new, rtol, atol, inv_sqrt_n)
        return (y_new, k7,
                jnp.maximum(max_abs, abs_norm),
                jnp.maximum(max_scaled, scaled))

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    # One matrix-vector product primes the FSAL carry; every later step
    # inherits its k1 from the previous step's seventh stage.
    carry = (y0, gen @ y0, zero, zero)
    y, _, max_abs, max_scaled = jax.lax.fori_loop(0, n_steps, body, carry)
    return y, max_abs, max_scaled


def _adaptive_kernel(generator, psi, t_final, initial_dt, rtol, atol,
                     max_steps):
    """
    Integrates with error-driven adaptive step sizes.

    Args:
        generator: The generator as a matrix.
        psi: The initial state as a vector.
        t_final: The total evolution time, which must be positive.
        initial_dt: The first step size to try.
        rtol: The relative tolerance.
        atol: The absolute tolerance.
        max_steps: The largest number of step attempts, accepted and rejected
            together, before giving up.

    Returns:
        tuple: The final state, the time reached, the accepted and rejected
            step counts, and the largest accepted per-step absolute and scaled
            error norms.
    """
    gen = jnp.asarray(generator, dtype=jnp.complex128)
    y0 = jnp.asarray(psi, dtype=jnp.complex128)
    inv_sqrt_n = 1.0 / jnp.sqrt(jnp.asarray(y0.size, dtype=jnp.float64))
    exponent = -1.0 / _ERROR_ORDER

    def cond(carry):
        t, _, _, _, n_accept, n_reject, _, _ = carry
        return (t < t_final) & (n_accept + n_reject < max_steps)

    def body(carry):
        t, y, k1, dt, n_accept, n_reject, max_abs, max_scaled = carry

        # Clamp the final step so the requested time is reached exactly, rather
        # than approached by accumulating rounding error.
        is_last = dt >= (t_final - t)
        h = jnp.where(is_last, t_final - t, dt)

        y_new, k7, error = _dp5_step(gen, y, k1, h)
        abs_norm = jnp.linalg.norm(error).astype(jnp.float64)
        scaled = _scaled_error_norm(error, y, y_new, rtol, atol, inv_sqrt_n)
        accept = scaled <= 1.0

        factor = _SAFETY * jnp.power(jnp.maximum(scaled, 1e-10), exponent)
        dt_next = dt * jnp.clip(factor, _MIN_FACTOR, _MAX_FACTOR)

        # On rejection y is unchanged, so its first stage k1 is still valid and
        # the FSAL carry survives the retry untouched.
        return (jnp.where(accept, jnp.where(is_last, t_final, t + h), t),
                jnp.where(accept, y_new, y),
                jnp.where(accept, k7, k1),
                dt_next,
                n_accept + jnp.where(accept, 1, 0),
                n_reject + jnp.where(accept, 0, 1),
                jnp.where(accept, jnp.maximum(max_abs, abs_norm), max_abs),
                jnp.where(accept, jnp.maximum(max_scaled, scaled), max_scaled))

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    carry = (zero, y0, gen @ y0, initial_dt,
             jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32),
             zero, zero)
    t, y, _, _, n_accept, n_reject, max_abs, max_scaled = \
        jax.lax.while_loop(cond, body, carry)
    return y, t, n_accept, n_reject, max_abs, max_scaled


def _get_fixed_kernel():
    """
    Returns the compiled fixed step kernel, compiling it on first use.
    """
    global _COMPILED_FIXED_KERNEL
    if _COMPILED_FIXED_KERNEL is None:
        _COMPILED_FIXED_KERNEL = jax.jit(_fixed_step_kernel)
    return _COMPILED_FIXED_KERNEL


def _get_adaptive_kernel():
    """
    Returns the compiled adaptive kernel, compiling it on first use.
    """
    global _COMPILED_ADAPTIVE_KERNEL
    if _COMPILED_ADAPTIVE_KERNEL is None:
        _COMPILED_ADAPTIVE_KERNEL = jax.jit(_adaptive_kernel)
    return _COMPILED_ADAPTIVE_KERNEL


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def rk45_evolve_with_error(psi: np.ndarray,
                           generator: np.ndarray,
                           time_difference: float,
                           rtol: float = 1e-10,
                           atol: float = 1e-10,
                           num_step: int = 1,
                           adaptive: bool = False,
                           max_steps: int = 10000,
                           device: Any = None
                           ) -> Tuple[np.ndarray, float, float]:
    """
    Solves dy/dt = G y on a GPU, reporting the embedded error estimate.

    Args:
        psi (np.ndarray): The initial state, as an arbitrary tensor. It is
            flattened for the solve and restored to its shape on return.
        generator (np.ndarray): The generator ``G`` as a matrix, with the sign
            and the imaginary unit of the evolution already applied.
        time_difference (float): The duration of the evolution. A negative
            duration evolves backwards; it is folded into the sign of the
            generator, which is exact for this linear problem.
        rtol (float, optional): The relative tolerance. Defaults to 1e-10,
            matching the scipy modes.
        atol (float, optional): The absolute tolerance. Defaults to 1e-10.
        num_step (int, optional): The number of equal steps to take, when
            stepping at a fixed size. Defaults to 1.
        adaptive (bool, optional): Whether to let the embedded error estimate
            choose the step size rather than stepping uniformly. Defaults to
            False.
        max_steps (int, optional): The step attempt budget for an adaptive
            solve. Defaults to 10000.
        device (optional): The device to run on. Defaults to the first GPU JAX
            can see. Naming one explicitly bypasses the GPU requirement, which
            is how the tests exercise this function off a GPU; production
            callers leave it unset.

    Returns:
        Tuple[np.ndarray, float, float]: The evolved state, the largest
            per-step error norm, and the largest per-step error norm scaled by
            ``rtol`` and ``atol``. A scaled value at or below one means every
            step met the requested tolerance.

    Raises:
        RuntimeError: If JAX cannot see a GPU, or if 64 bit mode is disabled.
        ValueError: If ``num_step`` is below one.
    """
    configure_jax()
    _require_x64()
    if device is None:
        device = require_gpu_device()

    psi = np.asarray(psi)
    out_shape = psi.shape
    size = psi.size
    time_difference = float(time_difference)

    if time_difference == 0.0:
        return (psi.astype(np.complex128, copy=True).reshape(out_shape),
                0.0, 0.0)

    gen = np.reshape(np.asarray(generator), (size, size))
    # A backward step arrives as a negative duration, which is how the TDVP
    # algorithms express their backward half-step. For dy/dt = G y, running
    # -G forwards is identical to running G backwards.
    if time_difference < 0.0:
        gen = -gen
        time_difference = -time_difference

    psi_flat = psi.reshape(-1)

    with jax.default_device(device):
        if adaptive:
            initial_dt = _estimate_initial_step_size(
                gen, psi_flat, time_difference, rtol, atol)
            result = _get_adaptive_kernel()(
                jnp.asarray(gen, dtype=jnp.complex128),
                jnp.asarray(psi_flat, dtype=jnp.complex128),
                jnp.asarray(time_difference, dtype=jnp.float64),
                jnp.asarray(initial_dt, dtype=jnp.float64),
                jnp.asarray(rtol, dtype=jnp.float64),
                jnp.asarray(atol, dtype=jnp.float64),
                jnp.asarray(int(max_steps), dtype=jnp.int32))
            y, t_reached, n_accept, n_reject, max_abs, max_scaled = result
            if int(n_accept) + int(n_reject) >= int(max_steps):
                warn('The adaptive JAX RK45 propagator exhausted its budget of '
                     f'{max_steps} step attempts and only reached t = '
                     f'{float(t_reached):.6g} of the requested '
                     f'{time_difference:.6g}. Loosen rtol and atol, or raise '
                     'max_steps.', RuntimeWarning, stacklevel=3)
        else:
            n_steps, dt = _step_plan(time_difference, num_step)
            y, max_abs, max_scaled = _get_fixed_kernel()(
                jnp.asarray(gen, dtype=jnp.complex128),
                jnp.asarray(psi_flat, dtype=jnp.complex128),
                jnp.asarray(dt, dtype=jnp.float64),
                jnp.asarray(n_steps, dtype=jnp.int32),
                jnp.asarray(rtol, dtype=jnp.float64),
                jnp.asarray(atol, dtype=jnp.float64))

    return (np.asarray(y).reshape(out_shape),
            float(max_abs), float(max_scaled))


def rk45_evolve(psi: np.ndarray,
                generator: np.ndarray,
                time_difference: float,
                rtol: float = 1e-10,
                atol: float = 1e-10,
                num_step: int = 1,
                adaptive: bool = False,
                max_steps: int = 10000,
                device: Any = None) -> np.ndarray:
    """
    Solves dy/dt = G y on a GPU via Dormand-Prince 5(4).

    Takes the same arguments as :func:`rk45_evolve_with_error`, but returns
    only the evolved state and warns if the embedded error estimate says the
    requested tolerance was missed.

    Returns:
        np.ndarray: The evolved state, with the shape of ``psi``.

    Warns:
        RuntimeWarning: If the worst step's scaled error estimate exceeds one,
            meaning the step size was too large for ``rtol`` and ``atol``.
    """
    state, max_abs, max_scaled = rk45_evolve_with_error(
        psi, generator, time_difference, rtol=rtol, atol=atol,
        num_step=num_step, adaptive=adaptive, max_steps=max_steps,
        device=device)
    if max_scaled > 1.0:
        warn('The JAX RK45 propagator exceeded the requested tolerance: the '
             f'worst step scaled error is {max_scaled:.3e} > 1 (absolute error '
             f'norm {max_abs:.3e}, rtol={rtol:.3e}, atol={atol:.3e}). Raise '
             "num_step, loosen rtol and atol, or pass adaptive=True in "
             'solver_options to let the solver choose its own step size.',
             RuntimeWarning, stacklevel=2)
    return state
