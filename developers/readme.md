# GPUMD Developer guide

* This file contains information useful for existing and future `GPUMD` developers.

* You can ignore this file if you are not interested in becoming a devoloper of `GPUMD`.

* This is a work in progress, which will be constantly updated along with the GPUMD developement activities.

## Using git to manage pull request (PR) contributions

* If you are new to this, here is a good place to start reading: https://git-scm.com/book/ms/v2/GitHub-Contributing-to-a-Project

## Development environments

* You can develop `GPUMD` in either Linux or Windows, as long as you have a working CUDA and/or HIP development toolkit, and a suitable GPU (Nvidia or AMD).

* GPUMD uses `make` to manage installation (or compilation).
We have not seen the necessity of using `cmake`, yet.

* There is no message passing interface (MPI) support in `GPUMD` yet, so currently you don't need to have MPI.
We might add MPI support in the future.

* To build `GPUMD`, simply type `make -f makefile` (for the CUDA version) or `make -f makefile.hip` (for the HIP version) and you will get the `gpumd` and `nep` binary files.

## External dependencies

* We make efforts to keep `GPUMD` as independent as possible.
In principle, we only use CUDA C++ and HIP C++ in the source code.
Particularly, we do not use Python in the source code.

* If you want to introduce external dependence, the relevant code must be made optional, which will not be compiled by default.
You alao need to give detailed instructions for setting up the necessary tools.

* Currently we have two external dependencies, the `NetCDF` package and the `PLUMED` package.

## Regression tests

* These are a few regression tests in the `tests` folder.
A developer should run the regression tests before starting a PR, saving the output files, and run the regression tests again after finishing the PR, making sure that the output files are identical.

## Source files

* All the source code of `GPUMD` can be found in the `src` folder.
They are either a source file with the `.cu` extension or a header file with the `.cuh` extension. 

* The source code was originally written in CUDA C++, but starting from GPUMD-v3.9.6, we support CUDA and HIP simultaneously.
The CUDA and HIP specific APIs are collected in `src/utilities/gpu_macro.cuh`.
If you use new CUDA and HIP APIs, they should be added to this file.

* We use `clang-format` to format all the source and header files, according to the specifications in the file `.clang-format`, which can be found in the main folder of the `GPUMD` package.

## Code structure

* `main_nep`: Starting point for neuroevolution potential (NEP) training.

* `main_gpumd`: Starting point for the `gpumd` executale. It contains the following modules:
  * `model`: The model dealing with the model system in the simulation. This module is used by all the other modules under `main_gpumd`.
  * `force`: The module contaning all the potentials.
  * `integrate`: The module containing all the integrators/ensembles.
  * `measure`: The module doing most of the on-the-fly calculations of the physical properties.
  * `phonon`: The module for doing phonon calculations.
  * `minimize`: The module for doing energy minimization.
  * `mc`: The module for doing hybrid Monte Carlo and molecular dynamics (MCMD) simulations.
* `utilities`: The module containing some common utilities used in many of other modules.

## Units

* Units for inputs and outputs should be specified in the user manual.

* Internally, we use the fwollowing basic units:
  * Energy: eV
  * Legnth: Angstrom
  * Mass: Dalton
  * Temperature: K
  * Charge: e (proton charge)

  
