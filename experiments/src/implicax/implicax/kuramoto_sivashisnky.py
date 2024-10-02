import jax
import jax.numpy as jnp

from .implicit_time_stepper import ImplicitTimeStepper
from .periodic_derivative_operators import PeriodicDerivatives1d

class KS(ImplicitTimeStepper):
    L: float
    N: int
    dt: float
    dx: float
    second_order_coeff: float
    fourth_order_coeff: float

    derivatives: PeriodicDerivatives1d


    def __init__(
        self,
        L: float,
        N: int,
        dt: float,
        *,
        second_order_coeff: float = 1.0,
        fourth_order_coeff: float = 1.0,
        crank_nicolson: bool = False,
        maxiter: int = 100,
        use_acceleration: bool = False,
        rev_method: str = "full",
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L / N

        self.second_order_coeff = second_order_coeff
        self.fourth_order_coeff = fourth_order_coeff

        self.derivatives = PeriodicDerivatives1d(L, N)

        super().__init__(
            crank_nicolson=crank_nicolson,
            use_acceleration=use_acceleration,
            maxiter=maxiter,
            adjoint_method=rev_method,
     
        )
    
    def _nonlinear_derivative_operator(self, u, *, linearize_at):
        return (
            self.derivatives.scaled_upwind_derivative__zero_fixed(u, winds=linearize_at)
            + self.second_order_coeff * self.derivatives.second_derivative_centered(u)
            + self.fourth_order_coeff * self.derivatives.fourth_derivative_centered(u)
        )
    