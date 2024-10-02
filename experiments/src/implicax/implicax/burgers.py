import jax
import jax.numpy as jnp

from .implicit_time_stepper import ImplicitTimeStepper
from .periodic_derivative_operators import PeriodicDerivatives1d, PeriodicDerivatives2d

class Burgers(ImplicitTimeStepper):
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
        restart: int = 20,
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
            self.derivatives.scaled_upwind_derivative__zero_fixed(u, winds=linearize_at)
            - self.nu * self.derivatives.second_derivative_centered(u)
        )
    
class Burgers2d(ImplicitTimeStepper):
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
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L / N
        self.nu = nu
        self.derivatives = PeriodicDerivatives2d(L, N)

        if solver == "direct":
            raise ValueError("Direct solver currently not supported for 2D.")

        super().__init__(
            crank_nicolson=crank_nicolson,
            use_acceleration=use_acceleration,
            maxiter_picard=maxiter_picard,
            maxiter_linsolve=maxiter_linsolve,
            adjoint_method=rev_method,
        )

    def _nonlinear_derivative_operator(self, u, *, linearize_at):
        winds_x = linearize_at[0:1]
        winds_y = linearize_at[1:2]

        vel_x = u[0:1]
        vel_y = u[1:2]

        applied_x = (
            self.derivatives.scaled_upwind_derivative__zero_fixed(vel_x, winds_x=winds_x, winds_y=winds_y)
            - self.nu * self.derivatives.laplace_centered(vel_x)
        )
        applied_y = (
            self.derivatives.scaled_upwind_derivative__zero_fixed(vel_y, winds_x=winds_x, winds_y=winds_y)
            - self.nu * self.derivatives.laplace_centered(vel_y)
        )
        applied = jnp.concatenate([applied_x, applied_y], axis=0)

        return applied
            