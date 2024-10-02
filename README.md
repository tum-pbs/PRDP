# PRDP
Progressively Refined Differentiable Physics

## Requirements
- Python 3.6
- Jax
- Jaxopt
- Equinox
- Optax
- Numpy
- Matplotlib
- ipykernel
- pdequinox
    - ```cd ./pdequinox && pip install -e .```

## How to run

Experiments are implemented as jupyter notebooks in the folder `experiments/`. 

List of experiments:
- `poisson_1_param.ipynb` - Inverse problem with 1 parameter
- `heat_1d.ipynb` - Linear neural emulator learning
- `heat_2d.ipynb` - Linear neural emulator learning
- `navier_stokes.ipynb` - Non-linear neural-hybrid corrector learning
