from src.linear_solvers_scan import forward_solve_SD, forward_solve_jacobi, forward_solve_b
import equinox as eqx
import jax
import jax.numpy as jnp

class BTCS_Stepper(eqx.Module):
    system_matrix: jax.Array
    n_iter_tracker: float
    jacobi_iter_max: int
    sd_iter_max: int
    
    def __init__(
        self,
        num_points: int,
        *,
        diffuse_amount: float = 0.001,
        n_iter_in = 1,
        dim = 1
    ):
        """
        Backward Time Centered Space (BTCS) stepper for the heat equation.
        num_points: number of grid points in each dimension
        """
        dx = 1 / (num_points + 1)
        if dim == 1:
            laplace_matrix = (
                -2 * jnp.diag(jnp.ones(num_points))
                + jnp.diag(jnp.ones(num_points - 1), k=1)
                + jnp.diag(jnp.ones(num_points - 1), k=-1)
            )
            laplace_matrix = laplace_matrix / dx**2
            self.system_matrix = jnp.eye(num_points) - diffuse_amount * laplace_matrix

        elif dim == 2:
            def matvec(x_flat):
                ndof = int(jnp.sqrt(jnp.squeeze(x_flat).shape[0]))
                x = jnp.reshape(x_flat, (ndof, ndof))
                x_padded = jnp.pad(x, ((1, 1), (1, 1)), mode='constant')
                x_new = x_padded[1:-1, 1:-1] - (0.001 / dx**2) * (
                    -4*x_padded[1:-1, 1:-1] + # center points 
                    x_padded[:-2, 1:-1] +  # left points
                    x_padded[2:, 1:-1] +   # right points
                    x_padded[1:-1, :-2] +  # bottom points
                    x_padded[1:-1, 2:]     # top points
                )
                return x_new.flatten()
            linearization_point = jnp.zeros(num_points**2)
            self.system_matrix = jax.jacfwd(matvec)(linearization_point)        

        else:
            raise ValueError("Only 1D and 2D systems are supported.")
        self.jacobi_iter_max = 50
        self.sd_iter_max = 20
        self.n_iter_tracker = n_iter_in

    # Direct Solver

    def __call__(self, state: jax.Array) -> jax.Array:
        return jnp.linalg.solve(self.system_matrix, state)
    
    # Jacobi

    def jacobi(self, state: jax.Array, u_init, n_iter) -> jax.Array:
        return forward_solve_jacobi(self.system_matrix, state, n_iter, u_init)[-1]
    
    def jacobi_dynamic(self, state: jax.Array, n_iterations = None, u_init = None) -> jax.Array:
        if n_iterations is None:
            n_iterations = int(self.n_iter_tracker)
            Warning(f"No number of iterations provided. Using {n_iterations} iterations.")
        if u_init is None:
            u_init = jnp.zeros_like(state)
            Warning("No initial guess provided. Using zero vector.")
        return forward_solve_jacobi(self.system_matrix, state, n_iterations, u_init)[-1]
    
    def jacobi_history(self, state: jax.Array, n_iterations: int, u_init) -> jax.Array:
        """Used for suboptimality plots"""
        return forward_solve_jacobi(self.system_matrix, state, n_iterations, u_init)
    
    def residuum_history(self, state: jax.Array, solver_name: str, n_iterations: int):
        primal_hist = forward_solve_b(
            self.system_matrix,
            state,
            n_iterations,
            solver_name,
            jnp.zeros_like(state)
        ) # shape (n_iterations, state_dim) = (n_iterations, N_dof)
        residuum_hist = jnp.dot(self.system_matrix, primal_hist.T) - state[:, None]
        return residuum_hist.T # shape (n_iterations, state_dim) = (n_iterations, N_dof)


    
    # Steepest Descent

    def sd(self, state: jax.Array, u_init = None) -> jax.Array:
        if u_init is None:
            u_init = jnp.zeros_like(state)
        return forward_solve_SD(self.system_matrix, state, self.sd_iter_max, u_init)[-1]
    
    def sd_dynamic(self, state: jax.Array, n_iterations = None, u_init = None) -> jax.Array:
        if n_iterations is None:
            n_iterations = self.n_iter_tracker
        if u_init is None:
            u_init = jnp.zeros_like(state)
            Warning("No initial guess provided. Using zero vector.")
        return forward_solve_SD(self.system_matrix, state, n_iterations, u_init)[-1]
    
    def sd_history(self, state: jax.Array, n_iterations, u_init) -> jax.Array:
        """Used for suboptimality plots"""
        return forward_solve_SD(self.system_matrix, state, n_iterations, u_init)
    
    # Factory for incrementing number of iterations

    def increment_n_iter(self, num_points: int, n_step: float, n_max: int = 40):
        """Increment the number of iterations by 1"""
        return BTCS_Stepper(num_points = num_points, 
                            n_iter_in = min(self.n_iter_tracker + n_step, n_max))
    
# =================================================================================================
## Generating batch of initial conditions
    
class TruncatedFourierSeries(eqx.Module):
    domain_extent: float
    sine_amplitudes: list[float]
    cosine_amplitudes: list[float]
    offset: float

    def __call__(self, x: jax.Array) -> jax.Array:
        u = sum(
            a_s * jnp.sin((i+1) * 2 * jnp.pi * x / self.domain_extent)
            +
            a_c * jnp.cos((i+1) * 2 * jnp.pi * x / self.domain_extent)
            for i, (a_s, a_c) in enumerate(zip(self.sine_amplitudes, self.cosine_amplitudes))
        )

        return u + self.offset
    
class TruncatedFourierSeries2D(eqx.Module):
    domain_extent: float
    sine_sine_amps: list[float]
    cosine_cosine_amps: list[float]
    sine_cosine_amps: list[float]
    cosine_sine_amps: list[float]
    offset: float

    def __call__(self, x: jax.Array) -> jax.Array:
        modes = []
        # for n, (a_s, a_c) in enumerate(zip(self.sine_sine_amps, self.cosine_amplitudes)):
        #     modes.append(a_s * jnp.sin((n+1)*2*jnp.pi * x[0] / self.domain_extent) * jnp.sin((n+1)*2*jnp.pi * x[1] / self.domain_extent)
        #                 +
        #                 a_c * jnp.cos((n+1)*2*jnp.pi * x[0] / self.domain_extent) * jnp.cos((n+1)*2*jnp.pi * x[1] / self.domain_extent))
        #     u = sum(modes)
        for n, (a_s_s, a_c_c, a_s_c, a_c_s) in enumerate(zip(self.sine_sine_amps, self.cosine_cosine_amps, self.sine_cosine_amps, self.cosine_sine_amps)):
            modes.append(a_s_s * jnp.sin((n+1)*2*jnp.pi * x[0] / self.domain_extent) * jnp.sin((n+1)*2*jnp.pi * x[1] / self.domain_extent)
                         +
                         a_c_c * jnp.cos((n+1)*2*jnp.pi * x[0] / self.domain_extent) * jnp.cos((n+1)*2*jnp.pi * x[1] / self.domain_extent)
                         +
                         a_c_s * jnp.cos((n+1)*2*jnp.pi * x[0] / self.domain_extent) * jnp.sin((n+1)*2*jnp.pi * x[1] / self.domain_extent)
                         +
                         a_s_c * jnp.sin((n+1)*2*jnp.pi * x[0] / self.domain_extent) * jnp.cos((n+1)*2*jnp.pi * x[0] / self.domain_extent)
                        )
            u = sum(modes)
        return u + self.offset
    
class RandomTruncatedFourierSeries(eqx.Module):
    """Randomly sample a truncated Fourier series."""
    domain_extent: float
    num_modes: int
    dim: int = 1
    amplitude_range: tuple[float, float] = (-1.0, 1.0)
    offset_range: tuple[float, float] = (0.0, 0.0)  # No offset by default

    def __call__(self, key) -> TruncatedFourierSeries:
        if self.dim == 1:
            sine_amp_key, cosine_amp_key, offset_key = jax.random.split(key, 3)
            sine_amplitudes = jax.random.uniform(
                sine_amp_key,
                (self.num_modes,),
                minval=self.amplitude_range[0],
                maxval=self.amplitude_range[1],
            )
            cosine_amplitudes = jax.random.uniform(
                cosine_amp_key,
                (self.num_modes,),
                minval=self.amplitude_range[0],
                maxval=self.amplitude_range[1],
            )
            offset = jax.random.uniform(
                offset_key,
                (1,),
                minval=self.offset_range[0],
                maxval=self.offset_range[1],
            )[0]
            return TruncatedFourierSeries(
                domain_extent=self.domain_extent,
                sine_amplitudes=sine_amplitudes,
                cosine_amplitudes=cosine_amplitudes,
                offset=offset,
            )
        elif self.dim == 2:
            c_c_key, s_s_key, s_c_key, c_s_key, offset_key = jax.random.split(key, 5)
            c_c_amplitudes = jax.random.uniform(
                c_c_key,
                (self.num_modes,),
                minval=self.amplitude_range[0],
                maxval=self.amplitude_range[1],
            )
            s_s_amplitudes = jax.random.uniform(
                s_s_key,
                (self.num_modes,),
                minval=self.amplitude_range[0],
                maxval=self.amplitude_range[1],
            )
            s_c_amplitudes = jax.random.uniform(
                s_c_key,
                (self.num_modes,),
                minval=self.amplitude_range[0],
                maxval=self.amplitude_range[1],
            )
            c_s_amplitudes = jax.random.uniform(
                c_s_key,
                (self.num_modes,),
                minval=self.amplitude_range[0],
                maxval=self.amplitude_range[1],
            )
            offset = jax.random.uniform(
                offset_key,
                (1,),
                minval=self.offset_range[0],
                maxval=self.offset_range[1],
            )[0]

            return TruncatedFourierSeries2D(
                domain_extent=self.domain_extent,
                sine_sine_amps=s_s_amplitudes,
                cosine_cosine_amps=c_c_amplitudes,
                sine_cosine_amps=s_c_amplitudes,
                cosine_sine_amps=c_s_amplitudes,
                offset=offset,
            )
        else:
            raise ValueError
    
def rollout(
    stepper: eqx.Module,
    num_steps: int,
    *,
    include_init: bool = False,
    solver_iterations = None
):
    """ 
    Given a stepper, return a function that takes an initial state and returns a trajectory
    Trajectory is of shape (num_steps, state_dim)
    """
    if solver_iterations is not None:
        def scan_fn(state, _):
            next_state = stepper(state, solver_iterations)
            return next_state, next_state
    else:
        def scan_fn(state, _):
            next_state = stepper(state)
            return next_state, next_state
    
    def rollout_fn(init_state):
        _, trj = jax.lax.scan(scan_fn, init_state, None, length=num_steps)

        if include_init:
            return jnp.concatenate([jnp.expand_dims(init_state, 0), trj], axis=0)
        else:
            return trj
        
    return rollout_fn


def dataloader(
    data,
    *,
    key,
    batch_size,
):
    n_samples = data.shape[0]

    n_batches = int(jnp.ceil(n_samples / batch_size))

    permutation = jax.random.permutation(key, n_samples)

    for batch_id in range(n_batches):
        start = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, n_samples)

        batch_indices = permutation[start:end]

        sub_data = data[batch_indices]

        yield sub_data