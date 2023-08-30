# Finite Element Stabilization for Flow Problems using SUPG 

<img src="img/supg.gif" width="720" height="360" />

## What's that?
In this code, we show how to use stabilized finite elements for flow problems. Two problems are solved:

- The scalar convection-diffusion 
- The incompressible Navier-Stokes equation


## How do we do that?
We use [FEniCS v.2019.1.0](https://fenicsproject.org/download/archive/) for finite element approximation and solvers

## Installation
This package is installed via conda ([Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)). Assuming conda is being executed, the following lines of code are enough for installing the package:

```make create_environment```

```conda activate supg```

```make env_to_kernel```

This allows the installation of all packages, including ipywidgets, required to run the Jupyter Notebook for the convection-diffusion problem. If you already have FEniCS installed, you can execute the scripts regardless the installation of this package.

