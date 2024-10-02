import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import jax.numpy as jnp
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


def plot_loss_and_gradient(value_history_vs_theta, gradient_history_vs_theta, x_values, title, theta_ref ):

    ITERATIONS_STEP = 20

    fig1, axs1 = plt.subplots(1,2)
    n_iter_max = value_history_vs_theta.shape[1]-1
    fig1.suptitle(title)

    for i in np.arange(0, n_iter_max, ITERATIONS_STEP):
        axs1[0].plot(x_values, value_history_vs_theta[:,i], lw=1, color=(i/n_iter_max, 0, 1-i/n_iter_max))#, label=f"Loss value at iteration {i}")
        axs1[1].plot(x_values, gradient_history_vs_theta[:,i], lw=1, color=(i/n_iter_max, 0, 1-i/n_iter_max))#, label=f"Loss gradient at iteration {i}")

    axs1[0].set_ylabel("Loss value (L2 norm of error)")
    axs1[1].set_ylabel("Loss gradient")

    # min_loss = min(values_list)
    # theta_min = x_values[values_list.index(min_loss)]
    # axs1[0].text(theta_min, min_loss, f"min loss = {min_loss:.2} \n at theta = {theta_min:.2}")

    for ax in axs1:
        ax.grid()
        ax.set_xlabel(r"$\theta$")
        ax.vlines(theta_ref, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='k', ls='--', lw=1, label=f"$\\theta$_ref = {theta_ref}")
        ax.legend()

    fig1.set_figheight(5)
    fig1.set_figwidth(15)
    fig1.text(s=f"Blue = 0 iterations", x=0.5, y=0.0, ha='center', va='bottom', color='b')
    fig1.text(s=f"Red = {n_iter_max} iterations", x=0.6, y=0.0, ha='center', va='bottom', color='r')


    # min_loss = min(values_list_2)
    # theta_min = x_values[values_list_2.index(min_loss)]
    # axs2[0].text(theta_min, min_loss, f"min loss = {min_loss:.2} \n at theta = {theta_min:.2}")
    return fig1

# def plot_loss_and_gradient_contours(value_history_vs_theta, gradient_history_vs_theta, x_values, title):
#     """
#     Plot the loss value as a contour plot

#     Inputs:
#         value_history_vs_theta: 2D array of loss values. 
#         Each row is a different value of theta, each column is a different iteration
    
#     """
#     assert(jnp.isnan(value_history_vs_theta).sum() == 0)
#     value_history_vs_theta = jnp.squeeze(value_history_vs_theta).T
#     gradient_history_vs_theta = jnp.squeeze(gradient_history_vs_theta).T
    
#     fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Gradient"), horizontal_spacing=0.15)

#     fig.add_trace(go.Contour(z=value_history_vs_theta, x=x_values, coloraxis = "coloraxis1"),row=1, col=1)

#     fig.add_trace(go.Contour(z=gradient_history_vs_theta, x=x_values, coloraxis = "coloraxis2", contours_showlabels=True), row=1, col=2)

#     fig.update_layout(  coloraxis =dict(colorbar_x=0.43),
#                         coloraxis2=dict(colorbar_x=1.0075)
#     )

#     fig.update_layout(title=dict(text=title, x=0.6, y=0.9, xanchor='center', yanchor='top'),)
#     fig.update_layout(
#         font = dict(family="Arial", size=14, color="black"),
#         width=1000,
#         height=500,
#         # autosize=False,
#         # margin=dict(l=65, r=50, b=65, t=90),
#         # paper_bgcolor="LightSteelBlue",
#     )
#     fig.update_xaxes(title_text = "theta")
#     fig.update_yaxes(title_text = "# inner iterations", row=1,col=1)

#     return fig


def plot_loss_and_gradient_3(value_history_vs_theta, gradient_history_vs_theta, x_values, title, theta_ref, ITERATIONS_STEP = 30 ):

    fig, axs = plt.subplots(1,4)
    fig.set_figheight(5)
    fig.set_figwidth(27)
    n_iter_max = value_history_vs_theta.shape[1]-1
    fig.suptitle(title)

    for i in np.arange(0, n_iter_max, ITERATIONS_STEP):
        axs[0].plot(x_values, value_history_vs_theta[:,i],      lw=1, color=(i/n_iter_max, 0, 1-i/n_iter_max))#, label=f"Loss value at iteration {i}")
        axs[1].plot(x_values, gradient_history_vs_theta[:,i,0], lw=1, color=(i/n_iter_max, 0, 1-i/n_iter_max))#, label=f"Loss gradient at iteration {i}")
        axs[2].plot(x_values, gradient_history_vs_theta[:,i,1], lw=1, color=(i/n_iter_max, 0, 1-i/n_iter_max))
        axs[3].plot(x_values, gradient_history_vs_theta[:,i,2], lw=1, color=(i/n_iter_max, 0, 1-i/n_iter_max))


    axs[0].set_title("Loss value"); axs[0].set_xlabel(r"$\theta_1 = \theta_2 = \theta_3$")
    axs[1].set_title(f"Grad w.r.t. $\\theta_1$"); axs[1].set_xlabel(r"$\theta_1$")
    axs[2].set_title(f"Grad w.r.t. $\\theta_2$"); axs[2].set_xlabel(r"$\theta_2$")
    axs[3].set_title(f"Grad w.r.t. $\\theta_3$"); axs[3].set_xlabel(r"$\theta_3$")

    axs[1].vlines(theta_ref[0], ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='k', ls='--', lw=1, label=f"$\\theta_0$_ref = {theta_ref[0]}")
    axs[2].vlines(theta_ref[1], ymin=axs[2].get_ylim()[0], ymax=axs[2].get_ylim()[1], color='k', ls='--', lw=1, label=f"$\\theta_1$_ref = {theta_ref[1]}")
    axs[3].vlines(theta_ref[2], ymin=axs[3].get_ylim()[0], ymax=axs[3].get_ylim()[1], color='k', ls='--', lw=1, label=f"$\\theta_2$_ref = {theta_ref[2]}")
    # min_loss = min(values_list)
    # theta_min = x_values[values_list.index(min_loss)]
    # axs1[0].text(theta_min, min_loss, f"min loss = {min_loss:.2} \n at theta = {theta_min:.2}")

    for ax in axs:
        ax.grid()
        # ax.set_xlabel(r"$\theta$")
        ax.legend()
    fig.text(s=f"Blue = 0 iterations", x=0.5, y=0.0, ha='center', va='bottom', color='b')
    fig.text(s=f"Red = {n_iter_max} iterations", x=0.6, y=0.0, ha='center', va='bottom', color='r')

    # min_loss = min(values_list_2)
    # theta_min = x_values[values_list_2.index(min_loss)]
    # axs2[0].text(theta_min, min_loss, f"min loss = {min_loss:.2} \n at theta = {theta_min:.2}")
    return fig

def plot_jac_subopt_3(theta_ref, n_dof, implicit_jacobi, unrolled_jacobi, implicit_SD, unrolled_SD):
    no_of_plots = 2
    fig, axs = plt.subplots(2, no_of_plots)
    last_iter = -1
    fig.set_figwidth(10 * no_of_plots)
    fig.set_figheight(10)
    fig.suptitle("Jacobian Suboptimality", fontsize=16)

    # Plot jacobi's jacobian suboptimality
    axs[0,0].plot(unrolled_jacobi[:last_iter, 0],  label="unrolled, $\\theta_1$", color='b')
    axs[0,0].plot(unrolled_jacobi[:last_iter, 1],  label="unrolled, $\\theta_2$", color='r')
    axs[0,0].plot(unrolled_jacobi[:last_iter, 2],  label="unrolled, $\\theta_3$", color='g')
    axs[0,0].plot(implicit_jacobi[:last_iter, 0], '--', label="implicit, $\\theta_1$", color='b')
    axs[0,0].plot(implicit_jacobi[:last_iter, 1], '--', label="implicit, $\\theta_2$", color='r')
    axs[0,0].plot(implicit_jacobi[:last_iter, 2], '--', label="implicit, $\\theta_3$", color='g')
    axs[0,0].set_title(f"JACOBI \n $\\theta$ = {theta_ref}, n_dofs = {n_dof}")

    # Plot SD's jacobian suboptimality
    axs[0,1].plot(unrolled_SD[:last_iter, 0],  label="unrolled, $\\theta_1$", color='b')
    axs[0,1].plot(unrolled_SD[:last_iter, 1],  label="unrolled, $\\theta_2$", color='r')
    axs[0,1].plot(unrolled_SD[:last_iter, 2],  label="unrolled, $\\theta_3$", color='g')
    axs[0,1].plot(implicit_SD[:last_iter, 0], '--', label="implicit, $\\theta_1$", color='b')
    axs[0,1].plot(implicit_SD[:last_iter, 1], '--', label="implicit, $\\theta_2$", color='r')
    axs[0,1].plot(implicit_SD[:last_iter, 2], '--', label="implicit, $\\theta_3$", color='g')
    axs[0,1].set_title(f"SD \n $\\theta$ = {theta_ref}, n_dofs = {n_dof}")

    # norm of gradient wrt theta
    unrolled_grad_norm_jacobi = jnp.linalg.norm(unrolled_jacobi, axis=1)
    implicit_grad_norm_jacobi = jnp.linalg.norm(implicit_jacobi, axis=1)
    unrolled_grad_norm_SD     = jnp.linalg.norm(unrolled_SD, axis=1)
    implicit_grad_norm_SD     = jnp.linalg.norm(implicit_SD, axis=1)
    # Plot jacobi's jacobian suboptimality
    axs[1,0].plot(unrolled_grad_norm_jacobi[:last_iter],  label="unrolled")
    axs[1,0].plot(implicit_grad_norm_jacobi[:last_iter], '--', label="implicit")
    axs[1,0].set_title(f"Norm over $\\theta$")
    axs[1,1].plot(unrolled_grad_norm_SD[:last_iter],  label="unrolled")
    axs[1,1].plot(implicit_grad_norm_SD[:last_iter], '--', label="implicit")
    axs[1,1].set_title(f"Norm over $\\theta$")

    for ax in axs.flat:
        ax.grid()
        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Linear Solver Iteration")
        ax.set_ylabel("Jacobian Suboptimality")

    return fig

# def plot_contours(values, xtitle, ytitle, ztitle=None, z_log_flag=False, y_values=None, title = None):
#     fig = make_subplots(rows=1, cols=1) # subplot_titles=("Unrolling",),
    
#     if z_log_flag:
#         values = jnp.log10(values)

#     fig.add_trace(
#         go.Contour(z=values, y=y_values, 
#                    contours=dict(showlines=False, showlabels=True),
#                    colorbar=dict(title=ztitle, titleside='right', tickfont=dict(family="Arial", size=14, color="black"))),
#     )

#     if title != None: 
#         fig.update_layout(title=dict(text=title, x=0.6, y=0.9, xanchor='center', yanchor='top'),)
    
#     fig.update_yaxes(
#         title_text=ytitle, 
#         title_font=dict(family="Arial", size=14, color="black"),
#         tickfont=dict(family="Arial", size=14, color="black"),
#     )
#     fig.update_xaxes(
#         title_text=xtitle, 
#         title_font=dict(family="Arial", size=14, color="black"),
#         tickfont=dict(family="Arial", size=14, color="black"),
#         # type="log"
#     )
#     fig.update_layout(
#         font = dict(family="Arial", size=14, color="black"),
#         # width=2000,
#         # height=600,
#     )
    
#     fig.update_coloraxes(colorbar_title_text="Colorbar Title")

#     # fig.show()
#     return fig


def plot_outer_history(y_values, title, log_flag=False, iterations_step = 20):


    fig, ax = plt.subplots(figsize=(4,4))
    n_inner_max = y_values.shape[1]-1
    # creating color map
    norm = Normalize(vmin=0, vmax=n_inner_max)
    cmap = cm.get_cmap("plasma_r")

    # Plot all the but the last one
    for i in np.arange(0, y_values.shape[1]-1, iterations_step):
        ax.plot(y_values[:,i], linewidth=1, color=cmap(norm(i)))
    # Plot the last one
    ax.plot(y_values[:,-1], label=f"PRDP", color="green")

    if log_flag:
        ax.set_yscale('log')
    ax.set_ylabel(title)
    ax.grid()
    ax.set_xlabel("outer iteration")
    # axs_theta.vlines(theta_ref, ymin=axs_theta.get_ylim()[0], ymax=axs_theta.get_ylim()[1], color='k', ls='--', lw=1, label=f"$\\theta$_ref = {theta_ref}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', label="# inner iterations") 
    ax.legend(loc='lower right', bbox_to_anchor=(1.458, -0.16), frameon=True)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="# inner iterations")

    # fig_theta.set_size_inches(7, 5, forward=True)
    # fig_theta.suptitle(f"{title} vs Outer iteration", size="x-large")
    return fig

def plot_outer_history_2(y1_values, y2_values, y1_title=None, y2_title=None, log_flag=False, iterations_step = 50, suptitle=None, is_prdp_included=True):

    assert y1_values.shape[1] == y2_values.shape[1] # same number of inner iterations
    n_inner_max = y1_values.shape[1]

    # creating color map
    norm = Normalize(vmin=0, vmax=n_inner_max)
    cmap = cm.get_cmap("plasma_r")

    # Create the figure
    fig, axs = plt.subplots(1,2, figsize=(5.5, 3))
    
    for i in np.arange(0, n_inner_max, iterations_step):
        color = cmap(norm(i))
        axs[0].plot(y1_values[:,i],  linewidth=1, color=color) # label=f"{i}"
        axs[1].plot(y2_values[:,i],  linewidth=1, color=color, label=f"{i}")
    if is_prdp_included:
        axs[0].plot(    y1_values[:,-1], linewidth=1.5, label=f"PR", color="green")
        axs[1].plot(    y2_values[:,-1], linewidth=1.5, label=f"PR", color="green")
    
    axs[0].set_ylabel(y1_title)
    axs[1].set_ylabel(y2_title)

    for ax in axs:
        if log_flag:
            ax.set_yscale('log')
        ax.set_xlabel("# outer iterations")
    axs[0].grid()
    axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')#, title="# inner iterations")

    # fig.set_size_inches(7, 5, forward=True)
    # fig.suptitle(f"{title} vs Outer iteration", size="x-large")
    fig.suptitle(suptitle)
    
    # Create a ScalarMappab;e for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # fig.colorbar(sm, ax=axs, orientation='vertical', label="# inner iterations", anchor=(25, 0))
    # fig.colorbar(sm, ax=axs, orientation='vertical', label="# inner iterations", anchor=(1, 0))
    # if is_prdp_included:
    #     axs[1].legend(loc='lower right', bbox_to_anchor=(1.7, -0.25), frameon=True)

    # axs[1].set_ylim(1e-6, 1e1)
    axs[1].minorticks_on()
    axs[1].grid(which='major', axis='both')
    axs[1].grid(which='minor', axis='x', linestyle=':', linewidth='0.5')

    fig.set_constrained_layout(True)

    return fig


# def add_scatter_to_contour(fig, n_inner_history):
#     fig.add_trace(go.Scatter(
#         x=n_inner_history, 
#         y=np.arange(n_inner_history.shape[0]), 
#         mode='lines', fillcolor='red', marker=dict(color="red"), showlegend=True,
#         name=r'dynamic N'), row=1, col=1)
#     return fig