import jax
from jax import Array
import jax.numpy as jnp

from .base_stepper import PicardStepper

class ImplicitTimeStepper(PicardStepper):
    """
    Simplifies the implementation of time steppers based on the Picard iteration
    for transient PDEs of the form

        du/dt + F(u) = 0
    
    where F is a **nonlinear** differential operator.

    Subclasses must implement the `_nonlinear_derivative_operator` method.
    """

    crank_nicolson: bool

    def __init__(
        self,
        *,
        crank_nicolson: bool,
        maxiter_picard: int = 100,
        maxiter_linsolve: int = 100,
        use_acceleration: bool = False,
        adjoint_method: str = "full",
        solver: str = "gmres",
        restart: int = 20,
    ):
        self.crank_nicolson = crank_nicolson

        super().__init__(
            use_acceleration=use_acceleration,
            maxiter_picard=maxiter_picard,
            maxiter_linsolve=maxiter_linsolve,
            adjoint_method=adjoint_method,
            solver=solver,
            restart=restart,
        )

    def _nonlinear_derivative_operator(
        self,
        u,
        *,
        linearize_at,
    ):
        """
        Given a PDE of the form

            du/dt + F(u) = 0

        this method the effect of the nonlinear differential operator F on the
        solution u at the given state u.

        In particular, we are interested in F(U) of the form

            F(u; U) = A(U) * u
        
        where A is a linear operator that depends (nonlinearly) on the
        linearization point U.

        Args:
            - `u`: The state at which to evaluate the nonlinear operator.
            - `linearize_at`: The point at which to linearize the operator.

        Returns:
            - `A(linearize_at) @ u`: The effect of the nonlinear operator on the
              state u when being linearized at the point `linearize_at`.
        """ 
         
        raise NotImplementedError("Must be implemented by subclass.")
    
    def _linearized_residuum(
        self,
        u_next: Array,
        linearize_at: Array,
        u_prev: Array
    ) -> Array:
        u_next_contribution = self._nonlinear_derivative_operator(
            u_next,
            linearize_at=linearize_at,
        )

        if self.crank_nicolson:
            u_prev_contribution = self._nonlinear_derivative_operator(
                u_prev,
                linearize_at=u_prev,
            )
            total_contribution = 0.5 * (u_next_contribution + u_prev_contribution)
        else:
            total_contribution = u_next_contribution

        residual = (u_next - u_prev) + self.dt * total_contribution

        return residual