import jax.numpy as jnp

L = 1   # length of domain

def A_matrix(n_dof):
    A_super_diag = jnp.diag(jnp.ones(n_dof-1), k=1)
    A_diag = jnp.diag(-2 * jnp.ones(n_dof))
    A_sub_diag = jnp.diag(jnp.ones(n_dof-1), k=-1)
    
    h = 1/(n_dof+1)
    factor = 1/(h*h)

    A = factor * (A_super_diag + A_diag + A_sub_diag)
    
    return A

# Problem: d2u/dx2 = -theta * sin(2 pi x)
def rhs_sine_fn(theta, x):
    return -theta * jnp.sin(2*jnp.pi * x)

def rhs_step_fn(theta, x): 
    return jnp.where(x < 0.5, -theta, theta)

def u_direct(theta_ref, x, A, rhs_fn):
    """Returns the direct solution of the linear system (therefore contains discretization error)"""
    rhs = rhs_fn(theta_ref, x)
    return jnp.linalg.solve(A, rhs)

def u_exact_PDE(theta_ref, x):
    """Returns the exact solution of the PDE assuming sine forcing function"""
    return theta_ref * jnp.sin(2*jnp.pi*x) / (2*jnp.pi)**2

# Problem: d2u/dx2 = -t1*sin(2 pi x) - t2*sin(4 pi x) - t3*sin(6 pi x)
def rhs_sine_fn_3(theta, x):
    return -theta[0] * jnp.sin(2*jnp.pi*x) - theta[1] * jnp.sin(4*jnp.pi*x) - theta[2] * jnp.sin(6*jnp.pi*x)

def u_exact_PDE_3(theta, x):
    """Returns the exact solution of the PDE assuming sine forcing function"""
    # return jnp.array((theta_ref[0] * jnp.sin(2*jnp.pi * x) + theta_ref[1] * jnp.sin(4*jnp.pi * x) + theta_ref[2] * jnp.sin(6*jnp.pi * x)) / (2*jnp.pi)**2)
    term1 = theta[0] * jnp.sin((2*jnp.pi) * x) / (2*jnp.pi)**2
    term2 = theta[1] * jnp.sin((4*jnp.pi) * x) / (4*jnp.pi)**2
    term3 = theta[2] * jnp.sin((6*jnp.pi) * x) / (6*jnp.pi)**2
    return jnp.array(term1 + term2 + term3)