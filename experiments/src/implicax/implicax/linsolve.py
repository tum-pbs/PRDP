import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from jax import jacfwd
from typing import Optional
import warnings

# from unrollable_linsolves_jax.bicgstab import BiCGStab as UnrollableBiCGStab
# from unrollable_linsolves_jax.gmres import GMRES as UnrollableGMRES

TOL = 1e-5

class BaseLinSolve(eqx.Module):
    pass


class __GMRES_Solve(BaseLinSolve):
    maxiter: Optional[int] = None
    rtol: float = 1e-5
    atol: float = 1e-5
    restart: int = 20

    def compute_solution(self, lin_fun, rhs):
        operator = lx.FunctionLinearOperator(
            lin_fun,
            rhs,
        )
        solution = lx.linear_solve(
            operator,
            rhs,
            lx.GMRES(
                max_steps=self.maxiter,
                rtol=self.rtol,
                atol=self.atol,
                restart=self.restart,
            ),
        )
        return solution

    def __call__(self, lin_fun, rhs):
        return self.compute_solution(lin_fun, rhs).value
    
    def diagnose(self, lin_fun, rhs):
        n_outer_iter = self.compute_solution(lin_fun, rhs).stats["num_steps"]
        n_total_iter = n_outer_iter * self.restart
        return n_total_iter


class GMRES_Solve(BaseLinSolve):
    maxiter: Optional[int] = None
    restart: int = 20
    solve_method: str = "batched"

    def __call__(self, lin_fun, rhs):
        """ Solve the linear system using jax.scipy.sparse.linalg.gmres."""
        x, _ = jax.scipy.sparse.linalg.gmres(
            lin_fun,
            rhs,
            maxiter=self.maxiter,
            solve_method=self.solve_method,
            tol=TOL,
            restart=self.restart,
        )
        return x

    def diagnose(self, lin_fun, rhs, restart=None, maxiter=None):
        """ Return all iterates of the GMRES algorithm."""
        if restart is None:
            restart = self.restart
        if maxiter is None:
            maxiter = self.maxiter
        unrollable_solver = UnrollableGMRES(restart=restart, maxiter=maxiter)
        x_history = unrollable_solver(lin_fun, rhs)
        return x_history

class BiCG_Stab_Solve(BaseLinSolve):
    maxiter: Optional[int] = None

    def __call__(self, lin_fun, rhs):
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            lin_fun,
            rhs,
            maxiter=self.maxiter,
            tol=TOL
        )
        return x

    def diagnose(self, lin_fun, rhs, maxiter=None):
        """ Return all iterates of the BiCG_Stab algorithm."""
        if maxiter is None:
            maxiter = self.maxiter
        solver = UnrollableBiCGStab(maxiter=maxiter)
        x_history = solver(lin_fun, rhs)
        return x_history

class CG_Solve(BaseLinSolve):
    maxiter: Optional[int] = None

    def __call__(self, lin_fun, rhs):
        x, _ = jax.scipy.sparse.linalg.cg(
            lin_fun,
            rhs,
            maxiter=self.maxiter,
            tol=TOL
        )
        return x

    def diagnose(self, lin_fun, rhs):
        raise NotImplementedError("jax.scipy.sparse.linalg.cg does not support diagnostics.")

class Direct_Solve(BaseLinSolve):
    def __call__(self, lin_fun, rhs):
        return_shape = rhs.shape
        matrix = jacfwd(lin_fun)(jnp.zeros(rhs.shape))
        rhs = rhs.flatten()
        matrix = matrix.reshape((rhs.size, rhs.size))
        x = jax.scipy.linalg.solve(matrix, rhs)
        return x.reshape(return_shape)