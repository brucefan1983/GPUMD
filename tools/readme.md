## Overview

The `tools` directory contains auxiliary scripts and utilities for tasks such as format conversion, data analysis, structure generation, etc. Hope you find them helpful, but you should use them with caution. If you have questions on a tool, you can try to contact the creator.

## Directory Structure

The tools are grouped into three categories for now:

| **Category**            | **Tools**                                                    |
| ----------------------- | ------------------------------------------------------------ |
| Format Conversion       | abacus2xyz, castep2exyz, cp2k2xyz, dp2xyz, exyz2pdb, gmx2exyz, mtp2xyz, orca2xyz, runner2xyz, vasp2xyz, xtd2exyz, xyz2gro |
| Analysis and Processing | add_groups, energy-reference-aligner, get_max_rmse_xyz, hydrogen_bond_analysis, pbc_mol, pca_sampling, perturbed2poscar, rdf_adf, select_xyz_frames, shift_energy_to_zero, split_xyz, |
| Miscellaneous           | doc_3.3.1, for_coding, md_tersoff, vim                       |

## Packages related to GPUMD and/or NEP:

Also, there are some packages that may be useful to you.

| Package        | link                                           | comment                                                      |
| -------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| `calorine`     | https://gitlab.com/materials-modeling/calorine | `calorine` is a Python package for running and analyzing MD simulations via GPUMD. It also provides functionality for constructing and sampling NEP models via GPUMD. |
| `GPUMD-Wizard` | https://github.com/Jonsnow-willow/GPUMD-Wizard | `GPUMD-Wizard` is a material structure processing software based on ASE (Atomic Simulation Environment) providing automation capabilities for calculating various properties of metals. Additionally, it aims to run and analyze MD simulations using GPUMD. |
| `gpyumd`       | https://github.com/AlexGabourie/gpyumd         | `gpyumd` is a Python interface for GPUMD. It helps users generate input and process output files based on the details provided by the GPUMD documentation. It currently supports up to GPUMD-v3.3.1 and only the gpumd executable. |
| `GPUMDkit`     | https://github.com/zhyan0603/GPUMDkit          | `GPUMDkit` is a toolkit for the GPUMD and NEP. It provides a set of tools to streamline the use of common scripts in GPUMD and NEP, simplifying workflows and enhancing efficiency. |
| `mdapy`        | https://github.com/mushroomfire/mdapy          | The `mdapy` Python library provides an array of powerful, flexible, and straightforward tools to analyze atomic trajectories generated from MD simulations. |
| `pynep`        | https://github.com/bigd4/PyNEP                 | `PyNEP` is a Python interface of the machine learning potential NEP used in GPUMD. |
| `somd`         | https://github.com/initqp/somd                 | `SOMD` is an ab-initio molecular dynamics (AIMD) package designed for the SIESTA DFT code. The SOMD code provides some common functionalities to perform standard Born-Oppenheimer molecular dynamics (BOMD) simulations, and contains a simple wrapper to the NEP package. The SOMD code may be used to automatically build NEPs by the mean of the active-learning methodology. |
| `NepTrain`     | https://github.com/aboys-cb/NepTrain           | An automated toolkit for training NEP, integrating tools like GPUMD, VASP, and NEP for streamlined workflows including perturbation, active learning, single-point energy calculations, and potential training. |
| `NepTrainKit`  | https://github.com/aboys-cb/NepTrainKit        | `NepTrainKit` is a Python package for visualizing and manipulating training datasets for NEP. |
| `NEP_Active`   | https://github.com/psn417/NEP_Active           | `NEP_Active` is a python package for building the training set using active learning strategy. It follows the method of Moment Tensor Potential (MTP). |
| `nep_maker`    | https://github.com/psn417/nep_maker            | This Python package facilitates the construction of NEP using active learning techniques. It employs the same strategies as MTP [J. Chem. Phys. 159, 084112 (2023)] and ACE [Phys. Rev. Materials 7, 043801 (2023)]. |

