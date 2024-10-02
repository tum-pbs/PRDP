from typing import Any
import jax
import jax.numpy as jnp
import jaxopt
import equinox as eqx

from jax import Array
from typing import Callable

class PeriodicDerivatives1d(eqx.Module):
    L: float
    N: int
    dx: float

    def __init__(
        self,
        L: float,
        N: int,
    ):
        self.L = L
        self.N = N
        self.dx = L / N

    def first_derivative_forward(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the forward difference of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1) - u) / self.dx
    
    def first_derivative_backward(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the backward difference of u given periodic boundary conditions.
        """
        return (u - jnp.roll(u, 1)) / self.dx
    
    def first_derivative_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the centered difference of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2 * self.dx)
    
    def second_derivative_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the second derivative of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / self.dx**2
    
    def fourth_derivative_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the fourth derivative of u given periodic boundary conditions.
        """
        return (
            jnp.roll(u, -2)
            - 4 * jnp.roll(u, -1)
            + 6 * u
            - 4 * jnp.roll(u, 1)
            + jnp.roll(u, 2)
        ) / self.dx**4
    
    def scaled_upwind_derivative__zero_fixed(
        self,
        u: Array,
        *,
        winds: Array,
    ) -> Array:
        """
        Computes the scaled upwind derivative of u given periodic boundary
        conditions and fixed winds.

        If applied in a Burgers equation setting, this should correctly
        propagate shocks.
        """
        positive_winds = jnp.maximum(
            (winds + jnp.roll(winds, 1)) / 2,
            0.0,
        )
        negative_winds = jnp.minimum(
            (winds + jnp.roll(winds, -1)) / 2,
            0.0,
        )

        upwind_diff = (
            positive_winds * self.first_derivative_backward(u)
            + negative_winds * self.first_derivative_forward(u)
        )

        return upwind_diff
    
class PeriodicDerivatives2d(eqx.Module):
    L: float
    N: int
    dx: float

    def __init__(
        self,
        L: float,
        N: int,
    ):
        self.L = L
        self.N = N
        self.dx = L / N

    def first_derivative_x_forward(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the forward difference of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1, axis=-1) - u) / self.dx
    
    def first_derivative_x_backward(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the backward difference of u given periodic boundary conditions.
        """
        return (u - jnp.roll(u, 1, axis=-1)) / self.dx
    
    def first_derivative_x_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the centered difference of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2 * self.dx)
    
    def first_derivative_y_forward(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the forward difference of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1, axis=-2) - u) / self.dx
    
    def first_derivative_y_backward(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the backward difference of u given periodic boundary conditions.
        """
        return (u - jnp.roll(u, 1, axis=-2)) / self.dx
    
    def first_derivative_y_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the centered difference of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2 * self.dx)
    
    def second_derivative_x_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the second derivative of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1, axis=-1) - 2 * u + jnp.roll(u, 1, axis=-1)) / self.dx**2
    
    def second_derivative_y_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the second derivative of u given periodic boundary conditions.
        """
        return (jnp.roll(u, -1, axis=-2) - 2 * u + jnp.roll(u, 1, axis=-2)) / self.dx**2
    
    def fourth_derivative_x_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the fourth derivative of u given periodic boundary conditions.
        """
        return (
            jnp.roll(u, -2, axis=-1)
            - 4 * jnp.roll(u, -1, axis=-1)
            + 6 * u
            - 4 * jnp.roll(u, 1, axis=-1)
            + jnp.roll(u, 2, axis=-1)
        ) / self.dx**4
    
    def fourth_derivative_y_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the fourth derivative of u given periodic boundary conditions.
        """
        return (
            jnp.roll(u, -2, axis=-2)
            - 4 * jnp.roll(u, -1, axis=-2)
            + 6 * u
            - 4 * jnp.roll(u, 1, axis=-2)
            + jnp.roll(u, 2, axis=-2)
        ) / self.dx**4
    
    def laplace_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the laplacian of u given periodic boundary conditions.
        """
        return self.second_derivative_x_centered(u) + self.second_derivative_y_centered(u)
    
    def double_laplace_centered(
        self,
        u: Array,
    ) -> Array:
        """
        Computes the laplacian of u given periodic boundary conditions.
        """
        return self.fourth_derivative_x_centered(u) + self.fourth_derivative_y_centered(u)
    
    def scaled_upwind_derivative__zero_fixed(
        self,
        u: Array,
        *,
        winds_x: Array,
        winds_y: Array,
    ) -> Array:
        """
        Computes the scaled upwind derivative of u given periodic boundary
        conditions and fixed winds.

        If applied in a Burgers equation setting, this should correctly
        propagate shocks.
        """
        positive_winds_x = jnp.maximum(
            (winds_x + jnp.roll(winds_x, 1, axis=-1)) / 2,
            0.0,
        )
        negative_winds_x = jnp.minimum(
            (winds_x + jnp.roll(winds_x, -1, axis=-1)) / 2,
            0.0,
        )
        positive_winds_y = jnp.maximum(
            (winds_y + jnp.roll(winds_y, 1, axis=-2)) / 2,
            0.0,
        )
        negative_winds_y = jnp.minimum(
            (winds_y + jnp.roll(winds_y, -1, axis=-2)) / 2,
            0.0,
        )

        upwind_diff_x = (
            positive_winds_x * self.first_derivative_x_backward(u)
            + negative_winds_x * self.first_derivative_x_forward(u)
        )
        upwind_diff_y = (
            positive_winds_y * self.first_derivative_y_backward(u)
            + negative_winds_y * self.first_derivative_y_forward(u)
        )

        return upwind_diff_x + upwind_diff_y
