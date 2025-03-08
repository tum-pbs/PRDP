# PRDP: Progressively Refined Differentiable Physics

Progressively Refined Differentiable Physics (PRDP) is a method to reduce the computational cost of learning pipelines (e.g. training of neural networks) that contain expensive iterative physics solvers. 

This repo provides the code and examples that accompany the paper linked below published in [ICLR 2025](https://iclr.cc/virtual/2025/poster/30710). The project page linked below provides a comprehensive overview of PRDP, including key concepts and visual explanations.

[📄 Paper](https://arxiv.org/abs/2502.19611)  • [🚀 Project Page](https://kanishkbh.github.io/prdp-paper/)


The implementation of PRDP is available in the file `experiments/src/prdp.py`.


## Requirements
- Python 3.10
- JAX
- Jaxopt
- Equinox
- Optax
- Numpy
- Matplotlib
- ipykernel
- lineax
- pdequinox
    - ```cd ./pdequinox && pip install -e .```

## How to run

Experiments are implemented as jupyter notebooks in the folder `experiments/`. 

List of experiments:
- `poisson_1_param.ipynb` - Inverse problem with 1 parameter
- `heat_1d.ipynb` - Linear neural emulator learning
- `heat_2d.ipynb` - Linear neural emulator learning
- `navier_stokes.ipynb` - Non-linear neural-hybrid corrector learning
