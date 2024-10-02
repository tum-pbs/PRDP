import jax
import jax.numpy as jnp

from typing import Tuple
from jax import Array

from .base_stepper import PicardStepper

class NavierStokesLidDriven(PicardStepper):
    L: float
    N: int
    dt: float
    nu: float
    lid_velocity: float

    dx: float

    _vel_x_tensor_shape: Tuple
    _vel_y_tensor_shape: Tuple
    _p_tensor_shape: Tuple

    _vel_x_dofs: int
    _vel_y_dofs: int
    _p_dofs: int

    _total_dofs: int

    def __init__(
        self,
        L: float,
        N: int,
        dt: float,
        *,
        nu: float,
        lid_velocity: float = 1.0,
        maxiter: int = 100,
        use_acceleration: bool = False,
        adjoint_method: str = "full",
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.nu = nu
        self.lid_velocity = lid_velocity

        self.dx = L / N
        self._vel_x_tensor_shape = (N+1, N)
        self._vel_y_tensor_shape = (N, N+1)
        self._p_tensor_shape = (N-1, N-1)

        self._vel_x_dofs = self._vel_x_tensor_shape[0] * self._vel_x_tensor_shape[1]
        self._vel_y_dofs = self._vel_y_tensor_shape[0] * self._vel_y_tensor_shape[1]
        self._p_dofs = self._p_tensor_shape[0] * self._p_tensor_shape[1]

        self._total_dofs = self._vel_x_dofs + self._vel_y_dofs + self._p_dofs

        super().__init__(
            use_acceleration=use_acceleration,
            maxiter=maxiter,
            adjoint_method=adjoint_method,
        )

    def _unpack(self, u):
        vel_x_component = u[0:self._vel_x_dofs]
        vel_y_component = u[self._vel_x_dofs:self._vel_x_dofs+self._vel_y_dofs]
        p_component = u[self._vel_x_dofs+self._vel_y_dofs:]

        vel_x = jnp.reshape(vel_x_component, self._vel_x_tensor_shape)
        vel_y = jnp.reshape(vel_y_component, self._vel_y_tensor_shape)
        p = jnp.reshape(p_component, self._p_tensor_shape)

        return vel_x, vel_y, p
    
    def _pack(self, vel_x, vel_y, p):
        u = jnp.concatenate([vel_x.flatten(), vel_y.flatten(), p.flatten()])
        return u
    
    def _laplace(self, field):
        """
        Only applies it to the interior
        """
        laplace_applied = jnp.zeros_like(field)
        laplace_applied = laplace_applied.at[1:-1, 1:-1].set(
            + field[0:-2, 1:-1]
            + field[2:, 1:-1]
            + field[1:-1, 0:-2]
            + field[1:-1, 2:]
            - 4 * field[1:-1, 1:-1]
        ) / (self.dx ** 2)
        return laplace_applied
    
    def _forward_diff_x(self, field):
        """ Only applies it to the interior """
        forward_diff_applied = jnp.zeros_like(field)
        forward_diff_applied = forward_diff_applied.at[1:-1, 1:-1].set(
            (field[2:, 1:-1] - field[1:-1, 1:-1]) / self.dx
        )
        return forward_diff_applied
    
    def _forward_diff_y(self, field):
        """ Only applies it to the interior """
        forward_diff_applied = jnp.zeros_like(field)
        forward_diff_applied = forward_diff_applied.at[1:-1, 1:-1].set(
            (field[1:-1, 2:] - field[1:-1, 1:-1]) / self.dx
        )
        return forward_diff_applied
    
    def _backward_diff_x(self, field):
        """ Only applies it to the interior """
        backward_diff_applied = jnp.zeros_like(field)
        backward_diff_applied = backward_diff_applied.at[1:-1, 1:-1].set(
            (field[1:-1, 1:-1] - field[0:-2, 1:-1]) / self.dx
        )
        return backward_diff_applied
    
    def _backward_diff_y(self, field):
        """ Only applies it to the interior """
        backward_diff_applied = jnp.zeros_like(field)
        backward_diff_applied = backward_diff_applied.at[1:-1, 1:-1].set(
            (field[1:-1, 1:-1] - field[1:-1, 0:-2]) / self.dx
        )
        return backward_diff_applied
    
    def _upwind_diff_x(self, field, vel_x_wind):
        """ Only applies it to the interior """
        positive_winds = jnp.maximum(vel_x_wind, 0)
        negative_winds = jnp.minimum(vel_x_wind, 0)

        upwind_diff_applied = (
            positive_winds * self._backward_diff_x(field)
            + negative_winds * self._forward_diff_x(field)
        )
        return upwind_diff_applied
    
    def _upwind_diff_y(self, field, vel_y_wind):
        """ Only applies it to the interior """
        positive_winds = jnp.maximum(vel_y_wind, 0)
        negative_winds = jnp.minimum(vel_y_wind, 0)

        upwind_diff_applied = (
            positive_winds * self._backward_diff_y(field)
            + negative_winds * self._forward_diff_y(field)
        )
        return upwind_diff_applied
    
    def map_vel_x_to_vel_y_grid(self, field):
        """ Only applies it to the interior """
        mapped = jnp.zeros(self._vel_y_tensor_shape)
        mapped = mapped.at[1:-1, 1:-1].set(
            field[1:-2, :-1]
            + field[1:-2, 1:]
            + field[2:-1, :-1]
            + field[2:-1, 1:]
        ) / 4
        return mapped
    
    def map_vel_y_to_vel_x_grid(self, field):
        """ Only applies it to the interior """
        mapped = jnp.zeros(self._vel_x_tensor_shape)
        mapped = mapped.at[1:-1, 1:-1].set(
            field[:-1, 1:-2]
            + field[1:, 1:-2]
            + field[:-1, 2:-1]
            + field[1:, 2:-1]
        ) / 4
        return mapped
    
    def _transient_effect(self, field, field_prev):
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
    
    def _pressure_gradient_effect__vel_x(self, p):
        """ Only applies it to the interior """
        pressure_gradient = jnp.zeros(self._vel_x_tensor_shape)
        pressure_gradient = pressure_gradient.at[1:-1, 1:-1].set(
            p[:, 1:] - p[:, :-1]
        ) / self.dx
        return pressure_gradient
    
    def _pressure_gradient_effect__vel_y(self, p):
        """ Only applies it to the interior """
        pressure_gradient = jnp.zeros(self._vel_y_tensor_shape)
        pressure_gradient = pressure_gradient.at[1:-1, 1:-1].set(
            p[1:, :] - p[:-1, :]
        ) / self.dx
        return pressure_gradient
    
    def _divergence_effect__from_vel_x(self, vel_x):
        """ Only applies it to the interior """
        divergence_contrib = (
            vel_x[1:-1, 1:] - vel_x[1:-1, :-1]
        ) / self.dx
        return divergence_contrib
    
    def _divergence_effect__from_vel_y(self, vel_y):
        """ Only applies it to the interior """
        divergence_contrib = (
            vel_y[1:, 1:-1] - vel_y[:-1, 1:-1]
        ) / self.dx
        return divergence_contrib
    
    def _bc_effect__vel_x(self, vel_x):
        bc_applied = vel_x
        bc_applied = bc_applied.at[0, :].set((vel_x[0, :] + vel_x[1, :]) / 2 - 0.0)

        # Enforces the lid-driven condition
        bc_applied = bc_applied.at[-1, :].set((vel_x[-1, :] + vel_x[-2, :]) / 2 - self.lid_velocity)
        
        return bc_applied
    
    def _bc_effect__vel_y(self, vel_y):
        bc_applied = vel_y
        bc_applied = bc_applied.at[:, 0].set((vel_y[:, 0] + vel_y[:, 1]) / 2 - 0.0)
        bc_applied = bc_applied.at[:, -1].set((vel_y[:, -1] + vel_y[:, -2]) / 2 - 0.0)
        return bc_applied
    
    def _momentum_effect__vel_x(self, vel_x, *, vel_x_prev, winds_x, winds_y):
        return (
            self._transient_effect(vel_x, vel_x_prev)
            +
            self._advection_effect__vel_x(vel_x, winds_x=winds_x, winds_y=winds_y)
            -
            self._diffusion_effect(vel_x)
        )
    
    def _momentum_effect__vel_y(self, vel_y, *, vel_y_prev, winds_x, winds_y):
        return (
            self._transient_effect(vel_y, vel_y_prev)
            +
            self._advection_effect__vel_y(vel_y, winds_x=winds_x, winds_y=winds_y)
            -
            self._diffusion_effect(vel_y)
        )
    
    def _linearized_residuum(
        self,
        u_next,
        linearize_at,
        u_prev,
    ):
        vel_x_next, vel_y_next, p_next = self._unpack(u_next)
        vel_x_prev, vel_y_prev, p_prev = self._unpack(u_prev)

        winds_x, winds_y, _ = self._unpack(linearize_at)

        vel_x_next__bc_applied = vel_x_next
        vel_y_next__bc_applied = vel_y_next

        vel_x_applied = (
            self._momentum_effect__vel_x(vel_x_next__bc_applied, vel_x_prev=vel_x_prev, winds_x=winds_x, winds_y=winds_y)
            +
            self._pressure_gradient_effect__vel_x(p_next)
        )
        vel_y_applied = (
            self._momentum_effect__vel_y(vel_y_next__bc_applied, vel_y_prev=vel_y_prev, winds_x=winds_x, winds_y=winds_y)
            +
            self._pressure_gradient_effect__vel_y(p_next)
        )
        p_applied = self._divergence_effect__from_vel_x(vel_x_next__bc_applied) + self._divergence_effect__from_vel_y(vel_y_next__bc_applied)

        u_applied = self._pack(
            self._bc_effect__vel_x(vel_x_applied),
            self._bc_effect__vel_y(vel_y_applied),
            p_applied,
        )
        return u_applied
    
    def get_vertex_collocated_velocities(
        self,
        u,
    ):
        vel_x_staggered, vel_y_staggerd, _ = self._unpack(u)
        vel_x = (
            vel_x_staggered[:-1, :]
            + vel_x_staggered[1:, :]
        ) / 2
        vel_y = (
            vel_y_staggerd[:, :-1]
            + vel_y_staggerd[:, 1:]
        ) / 2
        vel_collocated = jnp.stack(
            [vel_x, vel_y],
        )
        return vel_collocated

    
