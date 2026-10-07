"""Construction-time config checks shared by the BUG and RAGE integrators. Read-only."""

from warnings import warn

from .time_evolution import TimeEvoMode

__all__ = ["warn_nonhermitian_krylov", "validate_config"]


def warn_nonhermitian_krylov(solver) -> None:
    """Warn when ``config.hermitian=False`` is paired with a KRYLOV mode, whose fallback
    for a non-Hermitian generator is slower and not norm-preserving.

    An integrator whose generator is non-Hermitian by construction opts out by setting the
    class attribute ``_warn_nonhermitian_krylov = False``.

    Args:
        solver: The integrator; reads ``solver.config`` and the opt-out attribute.
    """
    if not getattr(solver, "_warn_nonhermitian_krylov", True):
        return
    if solver.config.local_solver()[0] is not TimeEvoMode.KRYLOV:
        return
    if not solver.config.hermitian:
        warn("config.hermitian=False with a KRYLOV mode: the Galerkin solver will use "
             "the general, non-symmetrised matrix-exponential Krylov path -- correct but "
             "slower, and the real-time flow is no longer norm-preserving. Use an "
             "EXPM/RK45 mode to avoid this path, or set hermitian=True if H_eff is in "
             "fact Hermitian (imaginary-time evolution of a Hermitian H is still "
             "hermitian=True).",
             UserWarning, stacklevel=3)


def validate_config(solver) -> None:
    """Check at construction that the config is the one this integrator takes.

    Raises:
        TypeError: The config is not an instance of the integrator's ``config_class``.
    """
    if not isinstance(solver.config, solver.config_class):
        raise TypeError(
            f"{type(solver).__name__} takes a {solver.config_class.__name__}, "
            f"not a {type(solver.config).__name__}.")
