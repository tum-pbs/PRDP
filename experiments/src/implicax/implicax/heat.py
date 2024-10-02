import jax
import jax.numpy as jnp

from .implicit_time_stepper import ImplicitTimeStepper
from .periodic_derivative_operators import PeriodicDerivatives1d, PeriodicDerivatives2d

class Heat(ImplicitTimeStepper):
    L: float
    N: int
    dt: float
    dx: float
    nu: float
    derivatives: PeriodicDerivatives1d

    def __init__(
        self,
        L: float,
        N: int,
        dt: float,
        *,
        nu: float = 0.01,
        crank_nicolson: bool = False,
        maxiter_picard: int = 100,
        maxiter_linsolve: int = 100,
        use_acceleration: bool = False,
        rev_method: str = "full",
        solver: str = "gmres",
        restart: int = 20, # for GMRES
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L / N
        self.nu = nu
        self.derivatives = PeriodicDerivatives1d(L, N)

        super().__init__(
            crank_nicolson=crank_nicolson,
            use_acceleration=use_acceleration,
            maxiter_picard=maxiter_picard,
            maxiter_linsolve=maxiter_linsolve,
            adjoint_method=rev_method,
            solver=solver,
            restart=restart,
        )

    def _nonlinear_derivative_operator(self, u, *, linearize_at):
        return (
            - self.nu * self.derivatives.second_derivative_centered(u)
        )

class Heat2d(ImplicitTimeStepper):
    L: float
    N: int
    dt: float
    dx: float
    nu: float
    derivatives: PeriodicDerivatives2d

    def __init__(
        self,
        L: float,
        N: int,
        dt: float,
        *,
        nu: float = 0.01,
        crank_nicolson: bool = False,
        maxiter_picard: int = 100,
        maxiter_linsolve: int = 100,
        use_acceleration: bool = False,
        rev_method: str = "full",
        solver: str = "gmres",
        restart: int = 20, # for GMRES
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L / N
        self.nu = nu
        self.derivatives = PeriodicDerivatives2d(L, N)

        super().__init__(
            crank_nicolson=crank_nicolson,
            use_acceleration=use_acceleration,
            maxiter_picard=maxiter_picard,
            maxiter_linsolve=maxiter_linsolve,
            adjoint_method=rev_method,
            solver=solver,
            restart=restart,
        )

    def _nonlinear_derivative_operator(self, u, *, linearize_at):
        return (
            - self.nu * self.derivatives.laplace_centered(u)
        )