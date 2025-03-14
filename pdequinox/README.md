
<h1 align="center">
  <img src="docs/imgs/pdequinox_logo.png" width="120">
  <br>
    PDEquinox
  <br>
</h1>

<h4 align="center">PDE Emulator Architectures in <a href="https://github.com/patrick-kidger/equinox" target="_blank">Equinox</a>.</h4>


<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#background">Background</a> •
  <a href="#features">Features</a> •
  <a href="#boundary-conditions">Boundary Conditions</a> •
  <a href="#constructors">Constructors</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

<p align="center">
    <img width=600 src="docs/imgs/pdequinox_teaser.png">
</p>

## Installation

Clone the repository, navigate to the folder and install the package with pip:
```bash
pip install .
```

Requires Python 3.10+ and JAX 0.4.13+. 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).


## Quickstart

Train a UNet to become an emulator for the 1D Poisson equation.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # `pip install optax`
import pdequinox as pdeqx
from tqdm import tqdm  # `pip install tqdm`

force_fields, displacement_fields = pdeqx.sample_data.poisson_1d_dirichlet(
    key=jax.random.PRNGKey(0)
)

force_fields_train = force_fields[:800]
force_fields_test = force_fields[800:]
displacement_fields_train = displacement_fields[:800]
displacement_fields_test = displacement_fields[800:]

unet = pdeqx.arch.ClassicUNet(1, 1, 1, key=jax.random.PRNGKey(1))

def loss_fn(model, x, y):
    y_pref = jax.vmap(model)(x)
    return jnp.mean((y_pref - y) ** 2)

opt = optax.adam(3e-4)
opt_state = opt.init(eqx.filter(unet, eqx.is_array))

@eqx.filter_jit
def update_fn(model, state, x, y):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, new_state = opt.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss

loss_history = []
shuffle_key = jax.random.PRNGKey(151)
for epoch in tqdm(range(100)):
    shuffle_key, subkey = jax.random.split(shuffle_key)

    for batch in pdeqx.dataloader(
        (force_fields_train, displacement_fields_train),
        batch_size=32,
        key=subkey
    ):
        unet, opt_state, loss = update_fn(
            unet,
            opt_state,
            *batch,
        )
        loss_history.append(loss)
```
## Background

Neural Emulators are networks learned to efficienty predict physical phenomena,
often associated with PDEs. In the simplest case this can be a linear advection
equation, all the way to more complicated Navier-Stokes cases. If we work on
Uniform Cartesian grids* (which this package assumes), one can borrow plenty of
architectures from image-to-image tasks in computer vision (e.g., for
segmentation). This includes:

* Standard Feedforward ConvNets
* Convolutional ResNets ([He et al.](https://arxiv.org/abs/1512.03385))
* U-Nets ([Ronneberger et al.](https://arxiv.org/abs/1505.04597))
* Dilated ResNets ([Yu et al.](https://arxiv.org/abs/1511.07122), [Stachenfeld et al.](https://arxiv.org/abs/2112.15275))
* Fourier Neural Operators ([Li et al.](https://arxiv.org/abs/2010.08895))

It is interesting to note that most of these architectures resemble classical
numerical methods or at least share similarities with them. For example,
ConvNets (or convolutions in general) are related to finite differences, while
U-Nets resemble multigrid methods. Fourier Neural Operators are related to
spectral methods. The difference is that the emulators' free parameters are
found based on a (data-driven) numerical optimization not a symbolic
manipulation of the differential equations.

(*) This means that we essentially have a pixel or voxel grid on which space is
discretized. Hence, the space can only be the scaled unit cube $\Omega = (0,
L)^D$

## Features

* Based on [JAX](https://github.com/google/jax):
  * One of the best Automatic Differentiation engines (forward & reverse)
  * Automatic vectorization
  * Backend-agnostic code (run on CPU, GPU, and TPU)
* Based on [Equinox](https://github.com/patrick-kidger/equinox):
  * Single-Batch by design
  * Integration into the Equinox SciML ecosystem
* Agnostic to the spatial dimension (works for 1D, 2D, and 3D)
* Agnostic to the boundary condition (works for Dirichlet, Neumann, and periodic
  BCs)
* Composability
* Tools to count parameters and assess receptive fields

## Boundary Conditions

This package assumes that the boundary condition is baked into the neural
emulator. Hence, most components allow setting `boundary_mode` which can be
`"dirichlet"`, `"neumann"`, or `"periodic"`. This affects what is considered a
degree of freedom in the grid.

![](docs/imgs/three_boundary_conditions.svg)

Dirichlet boundaries fully eliminate degrees of freedom on the boundary.
Periodic boundaries only keep one end of the domain as a degree of freedom (This
package follows the convention that the left boundary is the degree of freedom). Neumann boundaries keep both ends as degrees of freedom.

## Constructors

There are two primary architectural constructors for Sequential and Hierarchical
Networks that allow for composability with the `PDEquinox` blocks.

### Sequential Constructor

![](docs/imgs/sequential_net.svg)

The squential network constructor is defined by:
* a lifting block $\mathcal{L}$
* $N$ blocks $\left \{ \mathcal{B}_i \right\}_{i=1}^N$
* a projection block $\mathcal{P}$
* the hidden channels within the sequential processing
* the number of blocks $N$ (one can also supply a list of hidden channels if they shall be different between blocks)

### Hierarchical Constructor

![](docs/imgs/hierarchical_net.svg)

The hierarchical network constructor is defined by:
* a lifting block $\mathcal{L}$
* The number of levels $D$ (i.e., the number of additional hierarchies). Setting $D = 0$ recovers the sequential processing.
* a list of $D$ blocks $\left \{ \mathcal{D}_i \right\}_{i=1}^D$ for
  downsampling, i.e. mapping downwards to the lower hierarchy (oftentimes this
  is that they halve the spatial axes while keeping the number of channels)
* a list of $D$ blocks $\left \{ \mathcal{B}_i^l \right\}_{i=1}^D$ for
  processing in the left arc (oftentimes this changes the number of channels,
  e.g. doubles it such that the combination of downsampling and left processing
  halves the spatial resolution and doubles the feature count)
* a list of $D$ blocks $\left \{ \mathcal{U}_i \right\}_{i=1}^D$ for upsamping,
  i.e., mapping upwards to the higher hierarchy (oftentimes this doubles the
  spatial resolution; at the same time it halves the feature count such that we
  can concatenate a skip connection)
* a list of $D$ blocks $\left \{ \mathcal{B}_i^r \right\}_{i=1}^D$ for
  processing in the right arc (oftentimes this changes the number of channels,
  e.g. halves it such that the combination of upsampling and right processing
  doubles the spatial resolution and halves the feature count)
* a projection block $\mathcal{P}$
* the hidden channels within the hierarchical processing (if just an integer is
  provided; this is assumed to be the number of hidden channels in the highest
  hierarchy.)

### Beyond Architectural Constructors

For completion, `pdequinox.arch` also provides a `ConvNet` which is a simple
feed-forward convolutional network. It also provides `MLP` which is a dense
networks which also requires pre-defining the number of resolution points.

## Related

Similar packages that provide a collection of emulator architectures are
[PDEBench](https://github.com/pdebench/PDEBench) and
[PDEArena](https://github.com/pdearena/pdearena). With focus on Phyiscs-informed
Neural Networks and Neural Operators, there are also
[DeepXDE](https://github.com/lululxvi/deepxde) and [NVIDIA
Modulus](https://developer.nvidia.com/modulus).

## License

MIT, see [here](LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler)

