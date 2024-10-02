import jax
import jax.numpy as jnp
from jax.numpy import linalg as LA
from jax import jacfwd, jit
from functools import partial

@jit
def squareloss_fn(solution_history, ground_truth):
    """Returns the loss history (squared L2) i.e. loss vs iteration number.
    solution_history: shape = (n_iterations, n_dof)
    ground_truth: shape = (n_dof,)
    returns: shape = (n_iterations,)
    """
    N = ground_truth.shape[0]
    return ( (LA.norm(solution_history - ground_truth, axis=1))**2 / (2*N) )
    # return jnp.mean((solution - ground_truth)**2 ) * 0.5

# now value_array means value vs #inner_iterations
def outer_optimisation(objective_history_fn, n_outer_iterations=400, outer_lr = 1.0, init=5.0, n_inner_iterations=1000):
    """
    Solve the outer problem using gradient descent. Returns all iterates
    including the initial point. Solves the problem in batch for multiple
    iterations of the inner problem. Returns a matrix of shape `(n_outer_iter +
    1, n_inner_iter + 1)`, denoting the value of the parameter at each outer and
    inner iteration.
    
    Arguments:
        - objective_history_fn: function that returns the loss history (squared L2) i.e. loss vs iteration number.
        - n_outer_iterations: the number of outer iterations
        - outer_lr: the learning rate of the outer optimization
        - init: the initial value of the parameter theta for the outer optimization
        - n_inner_iterations: the number of inner iterations
        
    Returns:
        theta_history: the iterates of the parameter. Shape: (n_outer_iterations + 1, n_inner_iterations + 1)
        loss_history: the iterates of the loss. Shape: (n_outer_iterations + 1, n_inner_iterations + 1)
    """
    # objective_inner_history_fn = lambda theta: squareloss_fn(forward_solve(theta, A, rhs_fn, x, n_iterations=n_inner_iterations, solver=solver, print_flag=False), u_direct(THETA_REFERENCE, x, A, rhs_fn))
    grad_inner_history_fn = jacfwd(objective_history_fn)

    def step(theta_inner_history):
        # The outer grad function returns a vector of shape (n_inner_iter + 1,)
        # containing the gradients based on the number of inner iterations. We
        # do this for all thetas in the batch, so we get a matrix of shape
        # (n_inner_iter + 1, n_inner_iter + 1). **Since we do not adapt the
        # number of inner iterations during the outer optimization**, we only
        # select the diagonal of this matrix.
        matrix_of_grads = jax.vmap(grad_inner_history_fn)(theta_inner_history)
        matrix_of_grads = jnp.squeeze(matrix_of_grads)
        grads = jnp.diag(matrix_of_grads)
        theta_next_vs_inner = theta_inner_history - outer_lr * grads
        return theta_next_vs_inner

    def scan_fn(theta, _):
        # theta is `theta` vs `#InnerIterations`
        theta_next = step(theta)
        # vmap and compute loss at the new theta
        loss_history_vmap = jax.vmap(objective_history_fn)
        matrix_of_losses  = loss_history_vmap(theta_next)
        losses = jnp.diag(matrix_of_losses)
        output = (theta_next, losses)
        return theta_next, output
    
    theta_innerhistory_init = jnp.full((n_inner_iterations+1,), init)
    _, history = jax.lax.scan(scan_fn, theta_innerhistory_init, None, length=n_outer_iterations) # history is a tuple of (theta_history, loss_history)

    # append initial values to beginning of the history
    theta_history = jnp.concatenate([theta_innerhistory_init[jnp.newaxis], history[0]])
    loss_init = objective_history_fn(init)
    loss_history = jnp.append(loss_init[jnp.newaxis, :], history[1], 0)

    return theta_history, loss_history


def outer_optimisation_3(objective_history_fn, init, n_outer, lr=2000, n_inner=1000):
    """
    Solve the outer problem using gradient descent. Returns all iterates
    including the initial point. Solves the problem in batch for multiple
    iterations of the inner problem. Returns a matrix of shape `(n_outer_iter +
    1, n_inner_iter + 1)`, denoting the value of the parameter  and loss at each 
    outer and inner iteration.

    Arguments:
        - objective_history_fn: function that returns the loss history (squared L2) i.e. loss vs iteration number.
        - init: the initial value of the parameter vector theta for the outer optimization (jnp.array of shape (theta_dim,))
        - n_outer: the number of outer iterations
        - lr: the learning rate of the outer optimization
        - n_inner: the maximum number of inner iterations

    Returns:
        theta_history: the iterates of the parameter. Shape: (n_outer_iterations + 1, n_inner_iterations + 1, theta_dim)
        loss_history: the iterates of the loss. Shape: (n_outer_iterations + 1, n_inner_iterations + 1)
    """
    grad_history_fn = jacfwd(objective_history_fn)
    
    def body_fn(carry, _):
        # unpack
        theta_history = carry
        # vmap and compute gradient, the step theta
        grad_history_vmap = jax.vmap(grad_history_fn)
        matrix_of_grads   = grad_history_vmap(theta_history)
        grad_history  = matrix_of_grads[jnp.arange(n_inner+1), jnp.arange(n_inner+1), :] # diagonals of the (n_inner+1, n_inner+1) shaped matrix of grads (each grad is a vector of shape theta_dim. Therefore grad_history is of shape (n_inner+1, theta_dim))
        theta_history_next = theta_history - lr * grad_history
        # vmap and compute loss at the new theta
        loss_history_vmap = jax.vmap(objective_history_fn)
        matrix_of_losses  = loss_history_vmap(theta_history_next)
        losses = jnp.diag(matrix_of_losses)
        # return
        output_tuple = (theta_history_next, losses) # theta_history_next: (n_inner+1, theta_dim), losses: (n_inner+1,)
        return theta_history_next, output_tuple

    theta_innerhistory_init = jnp.tile(init, (n_inner+1, 1))
    _, history = jax.lax.scan(f=body_fn, init=theta_innerhistory_init, xs=None, length=n_outer)
    
    # append initial values to beginning of the history
    theta_history = jnp.append(theta_innerhistory_init[jnp.newaxis,:], history[0], 0)
    loss_init = objective_history_fn(init)
    loss_history = jnp.append(loss_init[jnp.newaxis,:], history[1], 0)

    # theta_history = history[0]
    # theta_history = jnp.concatenate([theta_init[jnp.newaxis], theta_history])
    return theta_history, loss_history

def outer_optimisation_dynamic_inner(loss_history_fn, n_outer_iterations, lr, theta_init, n_inner_init, n_inner_max, n_step, min_theta_change):
    """
    Solve the outer problem using gradient descent. Returns all iterates
    including the initial point. Solves the problem for a fixed number of inner
    iterations, and increases the number of inner iterations if the relative
    change in theta is small. Returns a matrix of shape `(n_outer_iterations + 1,)`, 
    denoting the value of the parameter at each outer iteration.

    Arguments:
        - loss_history_fn: function that returns the loss history (squared L2) i.e. loss vs iteration number.
        - n_outer_iterations: the number of outer iterations
        - lr: the learning rate of the outer optimization
        - theta_init: the initial value of the parameter vector theta for the outer optimization (jnp.array of shape (theta_dim,))
        - n_inner_init: the initial number of inner iterations
        - n_inner_max: the maximum number of inner iterations
        - n_step: the number of inner iterations to increase by if the relative change in theta is small
        - min_theta_change: the minimum relative change in theta to increase the number of inner iterations

    Returns:
        theta_history: the iterates of the parameter. Shape: (n_outer_iterations + 1, theta_dim)
        loss_history: the iterates of the loss. Shape: (n_outer_iterations + 1,)
        n_inner_history: history of n_inner used at each outer iteration. Shape: (n_outer_iterations + 1,)
    """

    n_inner_init = int(n_inner_init)
    print(f"outer_optimisation_dynamic_inner::\n theta_init = {theta_init}\n n_inner_init = {n_inner_init}\n n_inner_max = {n_inner_max}\n n_step = {n_step}\n lr = {lr}\n n_outer_iterations = {n_outer_iterations}\n=======")

    # loss_history_fn = lambda theta: jnp.squeeze(squareloss_fn(
    #     forward_solve_jacobi(theta, A, rhs_sine_fn, x, n_iterations=n_inner_max), 
    #     u_direct(THETA_REFERENCE, x, A, rhs_sine_fn)
    #     ))

    grad_history_fn = jacfwd(loss_history_fn)

    def GD_step_fn(carry, a):

        # unpack carry
        theta_prev, n_inner_prev = carry
        n_inner_int = n_inner_prev.astype(int)

        # 1. apply GD step
        grad_value = grad_history_fn(theta_prev)[n_inner_int] # grad at current theta, at n_inner_prev
        theta_next = theta_prev - lr*grad_value
        
        # 2. measure loss and relative change in theta
        loss_next = loss_history_fn(theta_next)[n_inner_int] # loss at current theta, at n_inner_prev
        rel_change = LA.norm(theta_prev - theta_next) / LA.norm(theta_prev)
        
        # 3. modify n_inner: increase by 10 if theta change is small, else keep same
        # n_inner_new = jax.lax.cond(rel_change > min_theta_change, # and n_inner_prev < N_INNER_MAX, 
        #                         lambda _: n_inner_prev, 
        #                         lambda _: n_inner_prev+n_step, 
        #                         None)  
        n_inner_new = n_inner_prev + 5

        # return
        carry_tuple = (theta_next, n_inner_new)
        output_tuple = (theta_next, loss_next, n_inner_new) # shapes: (theta_dim,), (), ()
        return carry_tuple, output_tuple

    carry_init = (theta_init, n_inner_init) # shapes: (theta_dim,), ()
    _, history = jax.lax.scan(GD_step_fn, carry_init, None, length=n_outer_iterations)
    
    # append initialization to the beginning of the history
    theta_history = jnp.array([theta_init, *history[0]])
    loss_init = loss_history_fn(theta_init)[n_inner_init]
    loss_history = jnp.array([loss_init, *history[1]])
    n_inner_history = jnp.array([n_inner_init, *history[2]]).astype(jnp.uint32)  # converting to int because actually the floor int of n_inner is used in the GD loop

    return theta_history, loss_history, n_inner_history


def forward_solve(THETA, A, rhs_fn, x, n_iterations, solver, u_init, print_flag = True):
    """
    Solve the linear system Ax = b for x, where rhs_fn is a function of theta.

    Arguments:
        - THETA: the value of the parameter
        - A: the matrix of the linear system
        - rhs_fn: function that returns right-hand side of the linear system
        - x: the grid points
        - n_iterations: the number of iterations
        - solver: the name of the solver to use for the inner optimization (jacobi, GS, SD)
        - print_flag: whether to print the residual at the end of the computation.
    
    Returns:
        u_iterates: the iterates of the solution including the initial state (Shape: (n_iterations + 1, n_dof))
    """
    rhs = rhs_fn(THETA, x)
    
    def jacobi_scan(u_old, _):
        
        U = jnp.triu(A, k=1)
        L = jnp.tril(A, k=-1)
        D = jnp.diag(A)
        u = (rhs - jnp.dot(U + L, u_old)) / D

        return u, u # ("carryover", "accumulated")
    
    
    def gauss_seidel_scan(u_old, _):
        
        U = jnp.triu(A, k=1)
        L = jnp.tril(A, k=-1)
        D = jnp.diag(jnp.diag(A)) # creates a diagonal matrix

        u = LA.inv(D + L) @ (rhs - U@u_old) # change the inv to triangular solve

        return u, u # ("carryover", "accumulated")
    
    def steepest_descent_scan(u_old, _):
        
        r = rhs - A@u_old
        d = r
        alpha = jnp.dot(r.T, r) / jnp.dot(d.T, A@d)
        u = u_old + alpha*d
        
        return u, u
    
    def richardson_iteration_scan(u_old, _):
        r = rhs - A@u_old
        u = u_old - 0.0001*r
        return u, u
    
    if solver == "jacobi":
        scan_fn = jacobi_scan
    elif solver == "GS":
        scan_fn = gauss_seidel_scan
    elif solver == "SD":
        scan_fn = steepest_descent_scan
    elif solver == "richardson":
        scan_fn = richardson_iteration_scan
    else:
        raise ValueError(f"unknown solver name: {solver}")
    
    _, u_iterates = jax.lax.scan(scan_fn, u_init, None, length=n_iterations)
    
    # append initial state to the beginning of the iterates
    u_iterates = jnp.concatenate((jnp.expand_dims(u_init, axis=0), u_iterates))
    
    # compute and print residual from the last iterate
    residual = LA.norm(jnp.dot(A, u_iterates[-1]) - rhs)
    if (print_flag): print(f"{solver}[{n_iterations}] residual = {residual} ")
    
    return u_iterates

# @partial(jit, static_argnums=(2,3))
def forward_solve_b(A, b, n_iterations, solver, u_init, print_flag = False):
    """
    Solve the linear system Au = b for u, where b is a function of theta.
    (Pass the computed rhs vector, not the rhs function.)

    Arguments:
        - THETA: the value of the parameter
        - A: the matrix of the linear system
        - b: right-hand side of the linear system
        - x: the grid points
        - n_iterations: the number of iterations
        - solver: the name of the solver to use for the inner optimization (jacobi, GS, SD)
        - print_flag: whether to print the residual at the end of the computation.
    
    Returns:
        u_iterates: the iterates of the solution including the initial state (Shape: (n_iterations + 1, n_dof))
    """
    
    def jacobi_scan(u_old, _):
        
        U = jnp.triu(A, k=1)
        L = jnp.tril(A, k=-1)
        D = jnp.diag(A)
        u = (b - jnp.dot(U + L, u_old)) / D

        return u, u # ("carryover", "accumulated")
    
    
    def gauss_seidel_scan(u_old, _):
        
        U = jnp.triu(A, k=1)
        L = jnp.tril(A, k=-1)
        D = jnp.diag(jnp.diag(A)) # creates a diagonal matrix

        u = LA.inv(D + L) @ (b - U@u_old) # change the inv to triangular solve

        return u, u # ("carryover", "accumulated")
    
    def steepest_descent_scan(u_old, _):
        
        r = b - A@u_old
        d = r
        alpha = jnp.dot(r.T, r) / jnp.dot(d.T, A@d)
        u = u_old + alpha*d
        
        return u, u
    
    def richardson_iteration_scan(u_old, _):
        r = b - A@u_old
        u = u_old - 0.0001*r
        return u, u
    
    if solver == "jacobi":
        scan_fn = jacobi_scan
    elif solver == "GS":
        scan_fn = gauss_seidel_scan
    elif solver == "SD":
        scan_fn = steepest_descent_scan
    elif solver == "richardson":
        scan_fn = richardson_iteration_scan
    else:
        raise ValueError(f"unknown solver name: {solver}")
    
    _, u_iterates = jax.lax.scan(scan_fn, u_init, None, length=n_iterations)
    
    # append initial state to the beginning of the iterates
    u_iterates = jnp.concatenate((jnp.expand_dims(u_init, axis=0), u_iterates))
    
    # compute and print residual from the last iterate
    residual = LA.norm(jnp.dot(A, u_iterates[-1]) - b)
    if (print_flag): print(f"{solver}[{n_iterations}] residual = {residual} ")
    
    return u_iterates


# @partial(jit, static_argnums=2)
def forward_solve_SD(A, b, n_iterations, u_initial):
    """
    A: matrix
    b: rhs
    return: final iterate of linsolve(A, b)
    """
    # u_initial = jnp.zeros(b.shape)
    # u_initial = jax.random.normal(jax.random.PRNGKey(0), shape=b.shape)
    # u_initial = jnp.ones(b.shape)
    # u_initial = jnp.array(2.0 / (2*jnp.pi)**2) * jnp.sin((2*jnp.pi) * x) # initial guess = exact solution
    
    def steepest_descent_scan(u_old, _):
        r = b - A@u_old
        d = r
        alpha = jnp.dot(r, r) / jnp.dot(d.T@A, d)
        u = u_old + alpha*d
        return u, u
    
    _, u_iterates = jax.lax.scan(steepest_descent_scan, u_initial, None, length=n_iterations)
    # append initial state to the beginning of the iterates
    u_iterates = jnp.concatenate((jnp.expand_dims(u_initial, axis=0), u_iterates))
    return u_iterates


def forward_solve_jacobi(A, b, n_iterations, u_initial):
    """
    A: matrix
    b: rhs
    return: solution history of Au=b
    """
    # u_initial = jnp.zeros(b.shape)
    # u_initial = jax.random.normal(jax.random.PRNGKey(0), shape=b.shape)
    # u_initial = jnp.ones(b.shape)
    # u_initial = jnp.array(2.0 / (2*jnp.pi)**2) * jnp.sin((2*jnp.pi) * x) # initial guess = exact solution

    def jacobi_scan(u_old, _):
        U = jnp.triu(A, k=1)
        L = jnp.tril(A, k=-1)
        D = jnp.diag(A).astype(float) # creates a column vector
        
        u = (b - jnp.dot(U + L, u_old)) / D

        return u, u # ("carryover", "accumulated")
    
    _, u_iterates = jax.lax.scan(jacobi_scan, u_initial, None, length=n_iterations)
    u_iterates = jnp.concatenate((jnp.expand_dims(u_initial, axis=0), u_iterates))
    return u_iterates

# def forward_solve_SD(A, b, n_iterations):
#     """
#     A: matrix
#     b: rhs
#     return: solution history of Au=b
#     """
#     # print(f"type(b) = {type(b)}")
#     u_initial = jnp.zeros(b.shape)
#     # u_initial = jax.random.normal(jax.random.PRNGKey(0), shape=b.shape)
#     # u_initial = jnp.ones(b.shape)
#     # u_initial = jnp.array(2.0 / (2*jnp.pi)**2) * jnp.sin((2*jnp.pi) * x) # initial guess = exact solution

#     def steepest_descent_scan(u_old, _):
#         r = b - A@u_old
#         d = r
#         alpha = jnp.dot(r.T, r) / jnp.dot(d.T, A@d)
#         u = u_old + alpha*d
#         return u, u
    
#     _, u_iterates = jax.lax.scan(steepest_descent_scan, u_initial, None, length=n_iterations)
#     # append initial state to the beginning of the iterates
#     u_iterates = jnp.concatenate((jnp.expand_dims(u_initial, axis=0), u_iterates))
#     return u_iterates


def forward_solve_matrix(THETA, A_fn, rhs, x, n_iterations, solver, print_flag = True):
    A = A_fn(THETA, x)
    
    def jacobi_scan(u_old, _):
        
        U = jnp.triu(A, k=1)
        L = jnp.tril(A, k=-1)
        D = jnp.diag(A).reshape(-1,1) # creates a column vector
        
        u = (rhs - jnp.dot(U + L, u_old)) / D

        return u, u # ("carryover", "accumulated")
    
    
    def gauss_seidel_scan(u_old, _):
        
        U = jnp.triu(A, k=1)
        L = jnp.tril(A, k=-1)
        D = jnp.diag(jnp.diag(A)) # creates a diagonal matrix

        u = LA.inv(D + L) @ (rhs - U@u_old) # jnp.linalg.solve(D + L, rhs - U@u_old)

        return u, u # ("carryover", "accumulated")
    
    def steepest_descent_scan(u_old, _):
        
        r = rhs - A@u_old
        d = r
        alpha = jnp.dot(r.T, r) / jnp.dot(d.T, A@d)
        u = u_old + alpha*d
        
        return u, u
    
    def richardson_iteration_scan(u_old, _):
        r = rhs - A@u_old
        u = u_old - 0.0001*r
        return u, u
    
    if solver == "jacobi":
        scan_fn = jacobi_scan
    elif solver == "GS":
        scan_fn = gauss_seidel_scan
    elif solver == "SD":
        scan_fn = steepest_descent_scan
    elif solver == "richardson":
        scan_fn = richardson_iteration_scan
    else:
        raise ValueError(f"unknown solver name: {solver}")
    
    u_initial = jnp.zeros(x.shape)
    _, u_iterates = jax.lax.scan(scan_fn, u_initial, None, length=n_iterations)
    
    # append initial state to the beginning of the iterates
    u_iterates = jnp.concatenate((jnp.expand_dims(u_initial, axis=0), u_iterates))
    
    # compute and print residual from the last iterate
    residual = LA.norm(jnp.dot(A, u_iterates[-1]) - rhs)
    if (print_flag): print(f"{solver}[{n_iterations}] residual = {residual} ")
    
    return u_iterates
