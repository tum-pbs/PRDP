import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Complex, PyTree
from typing import Callable

def get_mesh(
        L: float,
        N: int,
        *,
        full_mesh: bool = False
    ) -> Float[Array, "n_dof"] | Float[Array, "n_dof+1"]:
    """
    Returns an array of N equally spaced points on the interval [0, L) if
    `full_mesh` is False, or of N + 1 equally spaced points on the interval [0,
    L] if `full_mesh` is True.

    If `full_mesh` is False, this corresponds to the mesh points of the degrees
    of freedom for periodic boundary conditions. One of the boundary points is
    redundant; by convention, we drop the last point.

    **Parameters:**
        - `L`: The length of the interval.
        - `N`: The number of mesh points.
        - `full_mesh`: Whether to include the last point in the mesh. If `True`,
            the mesh will have N + 1 points; if `False`, the mesh will have N
            points.
    
    **Returns:**
        - `mesh`: An array of N equally spaced points on the interval [0, L) if
            `full_mesh` is False, or of N + 1 equally spaced points on the
            interval [0, L] if `full_mesh` is True.
    
    **Info:**
        To have a mesh with offset `O` you can elementwise add `O` to the
        returned mesh.
    
    **Example:**
        ```python
        >>> m_dof = get_mesh(1.0, 10)
        >>> m_dof
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        >>> m_full = get_mesh(1.0, 10, full_mesh=True)
        >>> m_full
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
        >>> m_dof.shape
        (10,)
        >>> m_full.shape
        (11,)
        ```
    """
    if full_mesh:
        return jnp.linspace(0.0, L, N + 1, endpoint=True)
    else:
        return jnp.linspace(0.0, L, N, endpoint=False)

def get_time_levels(
        dt: float,
        N_t: int,
        *,
        include_init: bool = False
    ) -> Float[Array, "n_steps"] | Float[Array, "n_steps+1"]:
    """
    Returns an array of equally spaced time levels.

    `include_init` controls whether the initial time level is included.

    **Parameters:**
        - `dt`: The time step size.
        - `N_t`: The number of time levels.
        - `include_init`: Whether to include the initial time level.

    **Returns:**
        - `time_levels`: An array of equally spaced time levels. If
            `include_init` is `True`, the array will have shape `(N_t + 1,)`; if
            `include_init` is `False`, the array will have shape `(N_t,)`.
    """
    time_levels_with_init = jnp.arange(N_t + 1) * dt
    if include_init:
        return time_levels_with_init
    else:
        return time_levels_with_init[1:]

def wrap_bc(u: Float[Array, "n_dof"]) -> Float[Array, "n_dof+1"]:
    """
    Wraps the periodic boundary conditions around the array `u`.

    This can be used to plot the solution of a periodic problem on the full
    interval [0, L] by plotting `wrap_bc(u)` instead of `u`.

    **Parameters:**
        - `u`: The array to wrap, shape `(N,)`.

    **Returns:**
        - `u_wrapped`: The wrapped array, shape `(N + 1,)`.
    """
    return jnp.pad(u, (0, 1), mode="wrap")

def rollout(
        stepper: Callable[[Float[Array, "n_dof"],], Float[Array, "n_dof"]],
        n: int,
        include_init: bool = False
    ):
    """
    Returns a function that takes an initial condition `u_0` and produces the
    discrete trajectory by autoregressively (recursively) applying the stepper
    `n` times.

    **Parameters:**
        - `stepper`: A function that takes an array `u` and returns the next
            state `u_next`. Expected signature: `u_next = stepper(u)` with
            shapes `(N,)` and `(N,)` respectively.
        - `n`: The number of steps to take.
        - `include_init`: Whether to include the initial condition in the
            returned trajectory. If `True`, the returned trajectory will have
            shape `(n + 1, N)`; if `False`, the returned trajectory will have
            shape `(n, N)`.
    
    **Returns:**
        - `rollout_stepper`: A function with the signature `trajectory =
            rollout_stepper(u_0)` with shapes `u_0.shape == (N,)` and
            `trajectory.shape == (n + 1, N)` if `include_init` is `True`, or
            `trajectory.shape == (n, N)` if `include_init` is `False`.
    
    **See also:**
        `rollout_with_forcing`
    """

    def _call_fun(u, _):
        u_next = stepper(u)
        return u_next, u_next
    
    if include_init:
        def rollout_stepper(u_0: Float[Array, "n_dof"]) -> Float[Array, "n_steps+1 n_dof"]:
            _, trajectory = jax.lax.scan(_call_fun, u_0, None, length=n)
            return jnp.concatenate([jnp.expand_dims(u_0, axis=0), trajectory], axis=0)
    else:
        def rollout_stepper(u_0: Float[Array, "n_dof"]) -> Float[Array, "n_steps n_dof"]:
            _, trajectory = jax.lax.scan(_call_fun, u_0, None, length=n)
            return trajectory
    return rollout_stepper

def rollout_with_forcing(
        forced_stepper: Callable[[Float[Array, "n_dof"], Float[Array, "n_dof"]], Float[Array, "n_dof"]],
        n: int,
        include_init: bool = False,
    ):
    """
    Returns a function that takes an initial condition `u_0` and a forcing
    trajectory `f` and produces the discrete trajectory by autoregressively
    (recursively) applying the stepper `n` times.

    **Parameters:**
        - `forced_stepper`: A function that takes a state vector `u` and a
            forcing vector `f` and returns the next state `u_next`. Expected
            signature: `u_next = forced_stepper(u, f)` with shapes `((N,), (N,))
            -> (N,)`.
        - `n`: The number of steps to take.
        - `include_init`: Whether to include the initial condition in the
            returned trajectory. If `True`, the returned trajectory will have
            shape `(n + 1, N)`; if `False`, the returned trajectory will have
            shape `(n, N)`.

    **Returns:**
        - `rollout_stepper_with_forcing`: A function with the signature
            `trajectory = rollout_stepper_with_forcing(u_0, f)`. The shape of
            `u_0` is `(N,)`. The shape of `f` is either `(n, N)` (then `n` has
            to be exactly the same as the number of steps) or `(N,)` (then `f`
            is repeated `n` times). The shape of `trajectory` is `(n + 1, N)` if
            `include_init` is `True`, or `(n, N)` if `include_init` is `False`.

    **See also:**
        `rollout`
    """
    def _call_fun(u, f):
        u_next = forced_stepper(u, f)
        return u_next, u_next
    
    if include_init:
        def rollout_stepper_with_forcing(
                u_0: Float[Array, "n_dof"],
                f: Float[Array, "n_steps n_dof"] | Float[Array, "n_dof"],
            ) -> Float[Array, "n_steps+1 n_dof"]:
            if f.shape == u_0.shape:
                f = jnp.repeat(jnp.expand_dims(f, axis=0), n, axis=0)
            _, trajectory = jax.lax.scan(_call_fun, u_0, f, length=n)
            return jnp.concatenate([jnp.expand_dims(u_0, axis=0), trajectory], axis=0)
    else:
        def rollout_stepper_with_forcing(
                u_0: Float[Array, "n_dof"],
                f: Float[Array, "n_steps n_dof"] | Float[Array, "n_dof"],
            ) -> Float[Array, "n_steps n_dof"]:
            if f.shape == u_0.shape:
                f = jnp.repeat(jnp.expand_dims(f, axis=0), n, axis=0)
            _, trajectory = jax.lax.scan(_call_fun, u_0, f, length=n)
            return trajectory
    return rollout_stepper_with_forcing

def repeat(
        stepper: Callable[[Float[Array, "n_dof"],], Float[Array, "n_dof"]],
        n: int,
    ):
    """
    Returns a function that takes an initial condition `u_0` and autoregressively
    (recursively) applies the stepper `n` times. No trajectory is recorded, only
    the final state is returned.

    **Parameters:**
        - `stepper`: A function that takes an array `u` and returns the next
            state `u_next`. Expected signature: `u_next = stepper(u)` with
            shapes `(N,)` and `(N,)` respectively.
        - `n`: The number of steps to take.

    **Returns:**
        - `repeated_stepper`: A function with the signature `final_state =
            repeated_stepper(u_0)` with shapes `u_0.shape == (N,)` and
            `final_state.shape == (N,)`.

    **See also:**
        `repeat_with_forcing`
    """
    def repeated_stepper(u_0: Float[Array, "n_dof"]) -> Float[Array, "n_dof"]:
        final, _ = jax.lax.scan(lambda u, _: (stepper(u), None), u_0, None, length=n)
        return final
    return repeated_stepper

def repeat_with_forcing(
        forced_stepper: Callable[[Float[Array, "n_dof"], Float[Array, "n_dof"]], Float[Array, "n_dof"]],
        n: int,
    ):
    """
    Returns a function that takes an initial condition `u_0` and a forcing
    trajectory `f` and autoregressively (recursively) applies the stepper `n`
    times. No trajectory is recorded, only the final state is returned.

    **Parameters:**
        - `forced_stepper`: A function that takes a state vector `u` and a
            forcing vector `f` and returns the next state `u_next`. Expected
            signature: `u_next = forced_stepper(u, f)` with shapes `((N,), (N,))
            -> (N,)`.
        - `n`: The number of steps to take.

    **Returns:**
        - `repeated_stepper_with_forcing`: A function with the signature
            `final_state = repeated_stepper_with_forcing(u_0, f)`. The shape of
            `u_0` is `(N,)`. The shape of `f` is either `(n, N)` (then `n` has
            to be exactly the same as the number of steps) or `(N,)` (then `f`
            is repeated `n` times). The shape of `final_state` is `(N,)`.

    **See also:**
        `repeat`
    """
    def repeated_stepper_with_forcing(
            u_0: Float[Array, "n_dof"],
            f: Float[Array, "n_steps n_dof"] | Float[Array, "n_dof"],
        ) -> Float[Array, "n_dof"]:
        if f.shape == u_0.shape:
            f = jnp.repeat(jnp.expand_dims(f, axis=0), n, axis=0)
        final, _ = jax.lax.scan(lambda u, f: (forced_stepper(u, f), None), u_0, f, length=n)
        return final
    return repeated_stepper_with_forcing

def spectral_derivative(
        u: Float[Array, "n_dof"],
        *,
        L: float,
        order: int = 1,
    ) -> Float[Array, "n_dof"]:
    """
    Returns the spectral derivative of `u`.

    **Parameters:**
        - `u`: The array to differentiate, shape `(N,)`. The array is assumed to
            be the equidistantly spaced values of a function on the interval
            `[0, L)`. (Important: The last point is not included in the mesh!)
        - `L`: The length of the interval.
        - `order`: The order of the derivative. Defaults to 1.

    **Returns:**
        - `u_der`: The spectral derivative of `u`, shape `(N,)`.
    """
    N = u.shape[-1]
    wavenumbers = jnp.fft.rfftfreq(N, 1/N)
    derivative_operator = 1j * wavenumbers * 2 * jnp.pi / L
    u_hat = jnp.fft.rfft(u)
    u_der_hat = derivative_operator**order * u_hat
    u_der = jnp.fft.irfft(u_der_hat, n=N)
    return u_der

def substack_trj(
    trj: PyTree[Float[Array, "n_timesteps ..."]],
    n: int,
) -> PyTree[Float[Array, "n_sub_trjs n ..."]]:
    """
    Slice a trajectory into subtrajectories of length `n` and stack them
    together. Useful for rollout training neural operators with temporal mixing.

    !!! Note that this function can produce very large arrays.

    **Parameters:**
        - `trj`: The trajectory to slice. Expected shape: `(n_timesteps, ...)`.
        - `n`: The length of the subtrajectories. If you want to perform rollout
            training with k steps, note that `n=k+1` to also have an initial
            condition in the subtrajectories.

    **Returns:**
        - `sub_trjs`: The stacked subtrajectories. Expected shape: `(n_stacks,
            n, ...)`. `n_stacks` is the number of subtrajectories stacked
            together, i.e., `n_sub_trjs` if `n_sub_trjs != -1` and
            `n_timesteps - n + 1` otherwise.
    """
    n_time_steps = [l.shape[0] for l in jtu.tree_leaves(trj)]

    if len(set(n_time_steps)) != 1:
        raise ValueError(
            "All arrays in trj must have the same number of time steps in the leading axis"
        )
    else:
        n_time_steps = n_time_steps[0]

    if n > n_time_steps:
        raise ValueError(
            "n must be smaller than or equal to the number of time steps in trj"
        )

    n_sub_trjs = n_time_steps - n + 1

    sub_trjs = jtu.tree_map(
        lambda trj: jnp.stack([trj[i : i + n] for i in range(n_sub_trjs)], axis=0),
        trj,
    )

    return sub_trjs


def l2_norm(u, *, L: float, squared: bool = False):
    """
    Computes the discrete consistent counterpart of the L2 function norm.
    Essentially this a RMSE (MSE if squared=True) of the function values scaled
    by srqt(L) (or L if squared=True).

    This arises based on the trapezoidal rule for integration with
    implementation of the periodic boundary conditions.

    !!! note
        As such the RMSE would only be consistent with the functional L2 norm if
        the function is periodic and the mesh is uniform.
    
    **Info:**
        If l2_norm is used to compute a relative functional error, i.e.
        l2_norm(u - u_ref) / l2_norm(u_ref), then scaling by the domain length
        L is not necessary. L can be any value.
    """

    squared_l2_norm = L * jnp.mean(u**2)
    if squared:
        return squared_l2_norm
    else:
        return jnp.sqrt(squared_l2_norm)

def build_ic_set(
    ic_generator,
    *,
    dof_mesh: Array,
    n_samples: int,
    key,
    return_conv_shape: bool = False,
):
    def scan_fn(k, _):
        k, sub_k = jr.split(k)
        ic_function = ic_generator(sub_k)
        discrete_ic = ic_function(dof_mesh)
        return k, discrete_ic
    
    _, ic_set = jax.lax.scan(scan_fn, key, None, length=n_samples)

    if return_conv_shape:
        ic_set = jnp.expand_dims(ic_set, axis=1)

    return ic_set