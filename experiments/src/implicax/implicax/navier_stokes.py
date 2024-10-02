import jax
import jax.numpy as jnp

from typing import Tuple
from jax import Array

from .base_stepper import PicardStepper

class NavierStokes(PicardStepper):
    L: float
    N: int
    dt: float
    nu: float

    dx: float
    _tensor_shape: Tuple

    def __init__(
        self,
        L: float,
        N: int,
        dt: float,
        *,
        nu: float,
        maxiter_picard: int = 100,
        use_acceleration: bool = False,
        adjoint_method: str = "full",
        maxiter_linsolve: int = 100,
        solver: str = "gmres",
        restart: int = 20,
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.nu = nu

        self.dx = L / N
        self._tensor_shape = (3, N, N)

        super().__init__(
            use_acceleration=use_acceleration,
            maxiter_picard=maxiter_picard,
            adjoint_method=adjoint_method,
            maxiter_linsolve=maxiter_linsolve,
            solver=solver,
            restart=restart,
        )

    
    def _unpack(self, u):
        vel_x_component = u[0:1]
        vel_y_component = u[1:2]
        p_component = u[2:3]
        return vel_x_component, vel_y_component, p_component

    def _laplace(self, field):
        return (
            + jnp.roll(field, 1, axis=-1)
            + jnp.roll(field, -1, axis=-1)
            + jnp.roll(field, 1, axis=-2)
            + jnp.roll(field, -1, axis=-2)
            - 4 * field
        ) / (self.dx ** 2)

    def _forward_diff_x(self, field):
        return (jnp.roll(field, -1, axis=-1) - field) / self.dx

    def _forward_diff_y(self, field):
        return (jnp.roll(field, -1, axis=-2) - field) / self.dx

    def _backward_diff_x(self, field):
        return (field - jnp.roll(field, 1, axis=-1)) / self.dx

    def _backward_diff_y(self, field):
        return (field - jnp.roll(field, 1, axis=-2)) / self.dx
    
    def _upwind_diff_x(self, field, vel_x_wind):
        positive_winds = jnp.maximum(vel_x_wind, 0)
        negative_winds = jnp.minimum(vel_x_wind, 0)

        return (
            positive_winds * self._backward_diff_x(field)
            + negative_winds * self._forward_diff_x(field)
        )

    def _upwind_diff_y(self, field, vel_y_wind):
        positive_winds = jnp.maximum(vel_y_wind, 0)
        negative_winds = jnp.minimum(vel_y_wind, 0)

        return (
            positive_winds * self._backward_diff_y(field)
            + negative_winds * self._forward_diff_y(field)
        )
    
    def map_vel_x_to_vel_y_grid(self, field):
        return (
            field
            + jnp.roll(field, -1, axis=-1)
            + jnp.roll(field, 1, axis=-2)
            + jnp.roll(field, (-1, 1), axis=(-1, -2))
        ) / 4
    
    def map_vel_y_to_vel_x_grid(self, field):
        return (
            field
            + jnp.roll(field, 1, axis=-1)
            + jnp.roll(field, -1, axis=-2)
            + jnp.roll(field, (1, -1), axis=(-1, -2))
        ) / 4
    
    def _transient_effect(self, field, *, field_prev):
        return (field - field_prev) / self.dt
    
    def _diffusion_effect(self, field):
        return self.nu * self._laplace(field)
    
    def _advection_effect__vel_x(self, field, *, winds_x, winds_y):
        winds_x_on_x_grid = winds_x
        winds_y_on_x_grid = self.map_vel_y_to_vel_x_grid(winds_y)
        return (
            self._upwind_diff_x(field, winds_x_on_x_grid)
            + self._upwind_diff_y(field, winds_y_on_x_grid)
        )

    def _advection_effect__vel_y(self, field, *, winds_x, winds_y):
        winds_x_on_y_grid = self.map_vel_x_to_vel_y_grid(winds_x)
        winds_y_on_y_grid = winds_y
        return (
            self._upwind_diff_x(field, winds_x_on_y_grid)
            + self._upwind_diff_y(field, winds_y_on_y_grid)
        )
    
    def rhs_assembly(self, u_prev):
        vel_x_prev, vel_y_prev, p_prev = self._unpack(u_prev)

        rhs = jnp.concatenate([
            vel_x_prev,
            vel_y_prev,
            jnp.zeros_like(p_prev),
        ])

        return rhs
    
    def _momentum_effect__vel_x(self, vel_x, *, vel_x_prev, winds_x, winds_y):
        return (
            self._transient_effect(vel_x, field_prev=vel_x_prev)
            +
            self._advection_effect__vel_x(vel_x, winds_x=winds_x, winds_y=winds_y)
            -
            self._diffusion_effect(vel_x)
        )
    
    def _momentum_effect__vel_y(self, vel_y, *, vel_y_prev, winds_x, winds_y):
        return (
            self._transient_effect(vel_y, field_prev=vel_y_prev)
            +
            self._advection_effect__vel_y(vel_y, winds_x=winds_x, winds_y=winds_y)
            -
            self._diffusion_effect(vel_y)
        )

    def _pressure_gradient_effect__vel_x(self, p):
        return self._forward_diff_x(p)

    def _pressure_gradient_effect__vel_y(self, p):
        return self._forward_diff_y(p)

    def _divergence_effect__vel_x(self, vel_x):
        return self._backward_diff_x(vel_x)

    def _divergence_effect__vel_y(self, vel_y):
        return self._backward_diff_y(vel_y)
    
    def __momentum_effect(self, vel_x__and__vel_y, *, vel_x__and_vel_y_prev, linearize_at):
        vel_x = vel_x__and__vel_y[0:1]
        vel_y = vel_x__and__vel_y[1:2]
        vel_x_prev = vel_x__and_vel_y_prev[0:1]
        vel_y_prev = vel_x__and_vel_y_prev[1:2]

        winds_x, winds_y, _ = self._unpack(linearize_at)

        return jnp.concatenate([
            self._momentum_effect__vel_x(vel_x, vel_x_prev=vel_x_prev, winds_x=winds_x, winds_y=winds_y),
            self._momentum_effect__vel_y(vel_y, vel_y_prev=vel_y_prev, winds_x=winds_x, winds_y=winds_y),
        ])
    
    def __pressure_gradient_effect(self, p):
        return jnp.concatenate([
            self._pressure_gradient_effect__vel_x(p),
            self._pressure_gradient_effect__vel_y(p),
        ])

    def divergence_effect(self, vel_x__and__vel_y):
        vel_x = vel_x__and__vel_y[0:1]
        vel_y = vel_x__and__vel_y[1:2]

        return self._divergence_effect__vel_x(vel_x) + self._divergence_effect__vel_y(vel_y)
    
    def _linearized_residuum(
        self,
        u_next: Array,
        linearize_at: Array,
        u_prev: Array,
    ):
        vel_x_next, vel_y_next, p_next = self._unpack(u_next)
        vel_x_prev, vel_y_prev, _ = self._unpack(u_prev)
        winds_x, winds_y, _ = self._unpack(linearize_at)


        vel_x_applied = (
            self._momentum_effect__vel_x(vel_x_next, vel_x_prev=vel_x_prev, winds_x=winds_x, winds_y=winds_y)
            +
            self._pressure_gradient_effect__vel_x(p_next)
        )
        vel_y_applied = (
            self._momentum_effect__vel_y(vel_y_next, vel_y_prev=vel_y_prev, winds_x=winds_x, winds_y=winds_y)
            +
            self._pressure_gradient_effect__vel_y(p_next)
        )
        p_applied = self._divergence_effect__vel_x(vel_x_next) + self._divergence_effect__vel_y(vel_y_next)

        u_applied = jnp.concatenate([
            vel_x_applied,
            vel_y_applied,
            p_applied,
        ])

        return u_applied
    
    def make_incompressible(self, u):
        vel_x, vel_y, p = self._unpack(u)
        divergence = self.divergence_effect(u)

        p_next = self.linsolve(lambda p: self.divergence_effect(self.__pressure_gradient_effect(p)), divergence)

        pressure_gradient = self.__pressure_gradient_effect(p_next)

        vel_x_next = vel_x - pressure_gradient[0:1]
        vel_y_next = vel_y - pressure_gradient[1:2]

        return jnp.concatenate([
            vel_x_next,
            vel_y_next,
            p_next,
        ])
        
    
    def compute_curl(self, u):
        vel_x, vel_y, _ = self._unpack(u)
        return self._backward_diff_x(vel_y) - self._backward_diff_y(vel_x)