from typing import Any
import jax
import jax.numpy as jnp
import jaxopt
import equinox as eqx

from jax import Array
from typing import Callable

from .linsolve import BaseLinSolve, GMRES_Solve, Direct_Solve, CG_Solve, BiCG_Stab_Solve

class PicardStepper(eqx.Module):
    """
    Subclass your PDE stepper from this class and implement the _linearized_residuum
    method.

    It will be a callable PyTree and a dataclass.
    """

    use_acceleration: bool
    maxiter_picard: int
    _maxiter: int  # Is one fewer if using reduced_prev or reversed_prev, but still performs maxiter iterations in total
    maxiter_linsolve: int
    adjoint_method: str
    _unrolled_diff: bool

    linsolve: BaseLinSolve

    def __init__(
        self,
        *,
        use_acceleration: bool = False,
        maxiter_picard: int = 100,
        maxiter_linsolve: int = 100,
        solver: str = "gmres",
        restart: int = 20,
        adjoint_method: str = "full",
    ):
        """
        Instantiate a time stepper with an inner Picard iteration.

        The inner iteration will run until convergence or maxiter iterations
        have been performed; whatever comes first.

        Each Picard iteration requires a linear solve which by default is done
        using GMRES of `jax.scipy.sparse.linalg`.

        Args:
            - `use_acceleration`: Whether to use Anderson acceleration or not.
                This can greatly accelerate the fixed point convegence. Also
                check out the notes in
                [jaxopt](https://jaxopt.github.io/stable/fixed_point.html#anderson-acceleration).
            - `maxiter_picard`: Maximum number of Picard iterations to perform.
            - `maxiter_linsolve`: Maximum number of linear solver iterations to perform.
            - `solver`: The linear solver to use. Options are:
                - `"gmres"`: GMRES solver from `jax.scipy.sparse.linalg`. This is
                    the default.
                - `"cg"`: Conjugate Gradient solver from `jax.scipy.sparse.linalg`.
                - `"bicgstab"`: BiCGStab solver from `jax.scipy.sparse.linalg`.
                - `"direct"`: Direct solver from `jax.scipy.linalg`.
            - `adjoint_method`: Method to use for the backpropagation over the
              stepper. Options are:
                - `"full"`: Use the full implicit differentiation capabilities
                    of jaxopt. This is the default. If the primal pass
                    converged, **this will give the exact gradient**.
                - `"unrolled"`: Reversely differentiate the unrolled Picard
                    iteration. This requires taping the primal iterates, and has
                    a higher memory footprint. If the primal pass converged,
                    this is identical to the `"full"` method, and **will give
                    the exact gradient**. If the primal pass did not fully
                    converge, this will give somewhat of the correct gradient of
                    the unconverged computational graph. In general, prefer
                    implicit differentiation.
                - `"reduced"`: Perform implicit differentiation, but **ignore
                    matrix assembly derivatives**. Oftentimes, especially with
                    dense matrix assemblies, this is faster than the `"full"`
                    approach. The resulting gradient will be **inexact**. The
                    level of inaccuracy depends on the influence of the matrix
                    assembly on the root. For example, in case of the implicit
                    Burgers equation, the inaccuracy is higher for higher
                    Reynolds numbers. Oftentimes, this is a valid approximation.
                    Similar to other implicit differentiation methods, this
                    requires the primal pass to converge.
                - `"reduced_prev"`: Same as `"reduced"`, but the adjoint pass
                    is linearized at the second to last iterate of the primal
                    pass (before convergence or before reaching maxiter).
                    Theoretically, this should save one matrix assembly but is
                    not yet implemented this way. It should allow to
                    differentiate a Picard-1 method (`maxiter=1`) with the
                    adjoint matrix also being assembled around the previous
                    state.
                - `"cheap_reversed"`: **Instead of differentiating, revert the
                    physics**, this is experimental. It allows to exploit the
                    autodiff engine to become an *inverse engine*. This applies
                    a matrix-vector product instead of a matrix.T-vector
                    linsolve. Note, that based on the condition number, this
                    introduces high-frequency noise into the adjoint pass,
                    especially when applied autoregressively.
                - `"cheap_reversed_prev"`: Same as `"cheap_reversed"`, but the
                    adjoint pass is linearized at the second to last iterate of
                    the primal pass (before convergence or before reaching
                    maxiter). (See also `"reduced_prev"`.)
                - `"cut"`: Do not differentiate the physics at all.
                    Backpropagation over this solver returns zero. Use this to
                    emulate a non-differentiable third-party solver.
                - `"_full_own"`: Same as `"full"`, but with a manual
                    implementation that does not differentiate the optimality
                    criterion.
        """
        if adjoint_method not in [
            "full",
            "unrolled",
            "reduced",
            "reduced_prev",
            "cheap_reversed",
            "cheap_reversed_prev",
            "cut",
            "_full_own",
        ]:
            raise ValueError(f"Unknown adjoint_method {adjoint_method}")

        self.use_acceleration = use_acceleration

        self.maxiter_picard = maxiter_picard
        if adjoint_method in ["reduced_prev", "cheap_reversed_prev"]:
            # "*_prev" methods use one fewer iteration during the picard solve
            # to record the second to last iterate for their adjoint pass. After
            # that, they perform the remaining iteration.
            self._maxiter = maxiter_picard - 1
        else:
            self._maxiter = maxiter_picard

        self.maxiter_linsolve = maxiter_linsolve

        self.adjoint_method = adjoint_method

        if adjoint_method == "unrolled":
            self._unrolled_diff = True
        else:
            self._unrolled_diff = False

        if solver == "gmres":
            self.linsolve = GMRES_Solve(maxiter=maxiter_linsolve, restart=restart)
            # jax.debug.print("Initializing GMRES solver")
        elif solver == "cg":
            self.linsolve = CG_Solve(maxiter=maxiter_linsolve)
            # jax.debug.print("Initializing CG solver")
        elif solver == "bicgstab":
            self.linsolve = BiCG_Stab_Solve(maxiter=maxiter_linsolve)
            # jax.debug.print("Initializing BiCGStab solver")
        elif solver == "direct":
            self.linsolve = Direct_Solve()
            # jax.debug.print("Initializing direct solver")
        else:
            raise ValueError(f"Unknown solver {solver}")

    def _linearized_residuum(
        self,
        u_next: Array,
        linearize_at: Array,
        u_prev: Array,
    ) -> Array:
        """
        Promised to be linear in the first argument.

        should be of the form

            g(u, U, v) = A(U) @ u - b(v)
        
        with u being the next solution, U the linearization point and the v the
        previous solution

        ### High-Level Examples:

        1. BTCS method for the heat equation:

            g(u, U, v) = u - v - dt * nu * L @ u

            with:
                - u: next solution
                - U: linearization point (not needed because this is a linear
                  problem)
                - v: previous solution
                - dt: time step size
                - nu: diffusion coefficient
                - L: Laplace operator  (in 1d, e.g., the three-point stencil)

        2. Implicit FOU for the Burgers equation:

            g(u, U, v) = u - v + dt * (upwind(U) @ u - nu * L @ u)

            with:
                - u: next solution
                - U: linearization point (in the case of advection problems also
                  called winds)
                - v: previous solution
                - dt: time step size
                - upwind: upwind operator:
                    - U * forward_diff(u) if U < 0
                    - U * backward_diff(u) if U > 0
                - nu: diffusion coefficient
                - L: Laplace operator  (in 1d, e.g., the three-point stencil)
        """
        raise NotImplementedError("Must be implemented in subclass")
    
    def residuum(
        self,
        u_next: Array,
        u_prev: Array,
    ) -> Array:
        """
        Computes the nonlinear residuum.

        Args:
            - `u_next`: The next solution.
            - `u_prev`: The previous solution.

        Returns:
            The nonlinear residuum.
        """
        return self._linearized_residuum(
            u_next,
            u_next,
            u_prev,
        )
    
    def _rhs_assembly(
        self,
        u_prev: Array,
    ) -> Array:
        """
        Extract the right hand side of the linearized residuum. Given the
        _linearized_residuum is of the form

            g(u, U, v) = A(U)u - b(v)

        we get the rhs by

            b(v) = -g(0, ⋅, v)
        
        In other words, we set u = 0 and negate the result. The value of the
        linearization point U does not matter, because the linearized residuum
        is linear in u and hence multiplied by 0.

        We set it U = 0 for convenience, because it requires the input to be of
        a certain shape.
        """
        neg_rhs = self._linearized_residuum(
            jnp.zeros_like(u_prev),
            jnp.zeros_like(u_prev),
            u_prev,
        )
        rhs = - neg_rhs
        return rhs
    
    def _lin_fun_assembly(
        self,
        linearize_at: Array,
    ) -> Callable[[Array,], Array]:
        """
        Returns a linear function that applies the effect of A(U) to a given
        vector u. The linearization point U is fixed.

        Given the _linearized_residuum is of the form

            g(u, U, v) = A(U)u - b(v)

        we get the linear function by

            f(u) = A(U)u = g(u, U, 0)

        In other words, we set v = 0. The value of the linearization point is
        captured in the closure.

        Args:
            - `linearize_at`: The linearization point U.

        Returns:
            A linear function that applies the effect of A(U) to a given vector
            u. Has the signature `f(u: Array) -> Array`.
        """
        lin_fun = lambda u: self._linearized_residuum(
            u,
            linearize_at,
            jnp.zeros_like(linearize_at),
        )
        return lin_fun
        
    
    def _picard_step(
        self,
        u_current_iter: Array,
        rhs: Array,
    ) -> Array:
        """
        Perform one iteration of the Picard method.

        This consists of two steps:

        1. Re-Assemble the system matrix at the current iterate. (Here we will
           produce a new linear operator that applies the effect of the matrix)
        2. Solve the linear system at the rhs (this rhs is constant for all
           Picard iterations and contains the information of the previous time
           step)

        Args:
            - `u_current_iter`: The current iterate.
            - `rhs`: The right hand side of the linear system. This is constant
                for all Picard iterations.

        Returns:
            The next iterate.
        """
        lin_fun = self._lin_fun_assembly(u_current_iter)
        u_next_iter = self.linsolve(lin_fun, rhs)
        return u_next_iter
    
    def _picard_solve(
        self,
        u_prev: Array,
    ) -> Array:
        """
        Perform a Picard iteration until convergence or maxiter iterations have
        been performed; whatever comes first.

        The initial guess is the previous solution.

        Args:
            - `u_prev`: The previous solution.

        Returns:
            The next solution.
        """
        initial_guess = jax.lax.stop_gradient(u_prev)
        rhs = self._rhs_assembly(u_prev)
        
        if self.use_acceleration:
            iterator = jaxopt.AndersonAcceleration(
                fixed_point_fun=self._picard_step,
                maxiter=self._maxiter,
                implicit_diff=not self._unrolled_diff,
            )
        else:
            iterator = jaxopt.FixedPointIteration(
                fixed_point_fun=self._picard_step,
                maxiter=self._maxiter,
                implicit_diff=not self._unrolled_diff,
            )

        res = iterator.run(
            initial_guess,
            rhs,
        )
        u_next = res.params
        return u_next
    
    def _primal_step(
        self,
        u_prev: Array,
    ) -> Array:
        """
        The primal pass of a Picard time stepper is to resolve a fixed point
        problem.
        """
        return self._picard_solve(u_prev)
    
    def single_step(
            self,
            u_prev: Array,
        ) -> Array:
        rhs = self._rhs_assembly(u_prev)
        # u_prev_gradient_stop = jax.lax.stop_gradient(u_prev)
        lin_fun = self._lin_fun_assembly(u_prev)
        return self.linsolve(lin_fun, rhs)
    
    
    def step(
        self,
        u_prev: Array,
    ) -> Array:
        """
        Advance the solution by one time step.

        We dispatch different functions based on the chosen adjoint method.

        Args:
            - `u_prev`: The previous solution state.
        
        Returns:
            The next state.
        """
        # Will be augmented in __init__ depending on adjoint_method
        if self.adjoint_method == "full":
            # Use the full implicit diff capabilities of jaxopt
            _step = self._primal_step

        elif self.adjoint_method == "unrolled":
            _step = self._primal_step

        elif self.adjoint_method == "reduced":
            _step = jax.custom_vjp(self._primal_step)
            _step.defvjp(self._adjoint_reduced_forward, self._adjoint_reduced_backward)
            
        elif self.adjoint_method == "reduced_prev":
            _step = jax.custom_vjp(self._primal_step)
            _step.defvjp(self._adjoint_reduced_prev_forward, self._adjoint_reduced_prev_backward)

        elif self.adjoint_method == "cheap_reversed":
            _step = jax.custom_vjp(self._primal_step)
            _step.defvjp(self._adjoint_cheap_reversed_forward, self._adjoint_cheap_reversed_backward)

        elif self.adjoint_method == "cheap_reversed_prev":
            _step = jax.custom_vjp(self._primal_step)
            _step.defvjp(self._adjoint_cheap_reversed_prev_forward, self._adjoint_cheap_reversed_prev_backward)

        elif self.adjoint_method == "cut":
            _step = lambda u: jax.lax.stop_gradient(self._primal_step(u))

        elif self.adjoint_method == "_full_own":
            _step = jax.custom_vjp(self._primal_step)
            _step.defvjp(self._adjoint_full_own_forward, self._adjoint_full_own_backward)
        
        return _step(u_prev)
    
    def get_all_iterates(
        self,
        u_prev: Array,
        *,
        include_init: bool = False,
    ) -> Array:
        """
        IMPORTANT: This does not obey the selected adjoint method!!!

        Returns all iterates of the Picard iteration.

        Args:
            - `u_prev`: The previous solution.
            - `include_init`: Whether to include the initial guess in the
                returned array. By default, the initial guess is the previous
                solution. Default: False.

        Returns:
            An array of shape (n_iterates, *u_prev.shape) containing all
            iterates of the Picard iteration. `n_iterates` is `self.maxiter_picard`
            even if the iteration converged earlier. Beyond convergence, the
            iterates are not meaningful anymore.
        """

        rhs = self._rhs_assembly(u_prev)

        def scan_fn(u_current_iter, _):
            u_next_iter = self._picard_step(u_current_iter, rhs)
            return u_next_iter, u_next_iter
        
        _, all_iterates = jax.lax.scan(
            scan_fn,
            u_prev,
            None,
            length=self.maxiter_picard,
        )

        if include_init:
            all_iterates = jnp.concatenate([jnp.expand_dims(u_prev, axis=0), all_iterates], axis=0)

        return all_iterates

    def diagnose(
        self,
        u_prev: Array,
        *,
        max_n_iter: int = 1000,
    ) -> int:
        """
        IMPORTANT: This does not obey the selected adjoint method!!!

        Returns the number of iterations needed to converge the Picard
        iteration.

        Args:
            - `u_prev`: The previous solution.
            - `max_n_iter`: The maximum number of iterations to perform. ! This
              differs from `self.maxiter_picard`. Default is 1000. 

        Returns:
            The number of iterations needed to converge the Picard iteration.
        """
        initial_guess = jax.lax.stop_gradient(u_prev)
        rhs = self._rhs_assembly(u_prev)
        
        if self.use_acceleration:
            iterator = jaxopt.AndersonAcceleration(
                fixed_point_fun=self._picard_step,
                maxiter=max_n_iter,  # Different from self.maxiter
            )
        else:
            iterator = jaxopt.FixedPointIteration(
                fixed_point_fun=self._picard_step,
                maxiter=max_n_iter,  # Different from self.maxiter
            )

        res = iterator.run(
            initial_guess,
            rhs,
        )
        n_iterations_needed = res.state.iter_num
        
        return n_iterations_needed
    
    def diagnose_one_linsolve(
        self,
        u_prev: Array,
        # rhs: Array,
    ):
        """ Return all iterates of the linear solver, starting from zero init."""
        rhs = self._rhs_assembly(u_prev)
        lin_fun = self._lin_fun_assembly(u_prev)
        return self.linsolve.diagnose(lin_fun, rhs)

    def diagnose_linsolve(
        self,
        u_prev: Array,
    ):
        all_picard_iterates = self.get_all_iterates(u_prev, include_init=True)
        rhs = self._rhs_assembly(u_prev)
        n_linsolve_iterations = jax.vmap(self.diagnose_one_linsolve, in_axes=(0, None))(all_picard_iterates, rhs)
        return n_linsolve_iterations


    def __call__(
        self,
        u_prev: Array,
    ) -> Array:
        """
        Advance the solution by one time step.

        Args:
            - `u_prev`: The previous solution state.

        Returns:
            The next state.
        """
        # return self.step(u_prev)
        return self.single_step(u_prev)
    
    # Below follow methods that are relevant for the specific adjoint methods.
    # They are more technical and only relevant if the relevant method is
    # chosen.
        
    def _adjoint_reduced_forward(
        self,
        u_prev
    ):
        u_next = self._primal_step(u_prev)
        carry = (u_prev, u_next,)
        return u_next, carry
    
    def _adjoint_reduced_backward(
        self,
        carry,
        du_next,
    ):
        u_prev, u_next = carry
        lin_fun = self._lin_fun_assembly(u_next)
        _lin_fun_transposed = jax.linear_transpose(lin_fun, u_next)
        lin_fun_transposed = lambda u: _lin_fun_transposed(u)[0]
        # adjoint_variable = self._linsolve(lin_fun_transposed, du_next)
        adjoint_variable = self.linsolve(lin_fun_transposed, du_next)
        _, rhs_assembly_vjp = jax.vjp(self._rhs_assembly, u_prev)
        du_prev, = rhs_assembly_vjp(adjoint_variable)

        return (du_prev,)
    
    def _adjoint_reduced_prev_forward(
        self,
        u_prev,
    ):
        u_second_to_last = self._primal_step(u_prev)
        rhs = self._rhs_assembly(u_prev)
        u_next = self._picard_step(u_second_to_last, rhs)
        carry = (u_prev, u_second_to_last, )
        return u_next, carry
    
    def _adjoint_reduced_prev_backward(
        self,
        carry,
        du_next,
    ):
        u_prev, u_second_to_last = carry
        lin_fun_second_to_last = self._lin_fun_assembly(u_second_to_last)
        _lin_fun_second_to_last_transposed = jax.linear_transpose(lin_fun_second_to_last, u_second_to_last)
        lin_fun_second_to_last_transposed = lambda u: _lin_fun_second_to_last_transposed(u)[0]
        # adjoint_variable = self._linsolve(lin_fun_second_to_last_transposed, du_next)
        adjoint_variable = self.linsolve(lin_fun_second_to_last_transposed, du_next)
        _, rhs_assembly_vjp = jax.vjp(self._rhs_assembly, u_prev)
        du_prev, = rhs_assembly_vjp(adjoint_variable)

        return (du_prev,)

    
    def _adjoint_cheap_reversed_forward(
        self,
        u_prev
    ):
        u_next = self._primal_step(u_prev)
        carry = (u_prev, u_next,)
        return u_next, carry
    
    def _adjoint_cheap_reversed_backward(
        self,
        carry,
        du_next,
    ):
        u_prev, u_next = carry
        lin_fun = self._lin_fun_assembly(u_next)
        reversed_du_next = lin_fun(du_next)
        _, rhs_assembly_vjp = jax.vjp(self._rhs_assembly, u_prev)
        du_prev, = rhs_assembly_vjp(reversed_du_next)

        return (du_prev,)
    
    def _adjoint_cheap_reversed_prev_forward(
        self,
        u_prev,
    ):
        u_second_to_last = self._primal_step(u_prev)
        rhs = self._rhs_assembly(u_prev)
        u_next = self._picard_step(u_second_to_last, rhs)
        carry = (u_prev, u_second_to_last, )
        return u_next, carry
    
    def _adjoint_cheap_reversed_prev_backward(
        self,
        carry,
        du_next,
    ):
        u_prev, u_second_to_last = carry
        lin_fun_second_to_last = self._lin_fun_assembly(u_second_to_last)
        reversed_du_next = lin_fun_second_to_last(du_next)
        _, rhs_assembly_vjp = jax.vjp(self._rhs_assembly, u_prev)
        du_prev, = rhs_assembly_vjp(reversed_du_next)

        return (du_prev,)
    
    def _adjoint_full_own_forward(
        self,
        u_prev,
    ):
        u_next = self._primal_step(u_prev)
        carry = (u_prev, u_next,)
        return u_next, carry
    
    def _adjoint_full_own_backward(
        self,
        carry,
        du_next,
    ):
        u_prev, u_next = carry
        _, _full_diff_vjp_fun = jax.vjp(lambda u: self._lin_fun_assembly(u)(u), u_next)
        full_diff_vjp_fun = lambda u: _full_diff_vjp_fun(u)[0]
        # adjoint_variable = self._linsolve(full_diff_vjp_fun, du_next)
        adjoint_variable = self.linsolve(full_diff_vjp_fun, du_next)
        _, rhs_assembly_vjp = jax.vjp(self._rhs_assembly, u_prev)
        du_prev, = rhs_assembly_vjp(adjoint_variable)

        return (du_prev,)
    
