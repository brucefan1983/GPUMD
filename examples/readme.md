# Examples and tutorials of GPUMD

## List of examples (only the initial creator is listed)


| folder                                     | creator       | potential | description                                        |
| ---------------------------------------    | ------------- | --------- | ---------------------------------------------------|
| 01_Carbon_examples_for_JCP_2022_paper      | Penghua Ying  | NEP       | Some examples for Ref. [1] |
| 02_Carbon_density_of_states                | Zheyong Fan   | Tersoff   | Phonon density of states of graphene |
| 03_Carbon_thermal_transport_emd            | Zheyong Fan   | Tersoff   | Thermal transport in graphene from EMD |
| 04_Carbon_thermal_transport_nemd_and_hnemd | Zheyong Fan   | Tersoff   | Thermal transport in graphene from NEMD and NEMD |
| 05_Carbon_phonon_vibration_viewer          | Ting Liang    | Tersoff   | Visualizing the phonon modes in a type of diamond nanowire. |
| 06_Silicon_phonon_dispersion               | Zheyong Fan   | Tersoff   | Phonon dispersions of silicon.  |
| 07_Silicon_thermal_expansion               | Zheyong Fan   | Tersoff   | Thermal expansion of silicon based on classical MD. |
| 08_Silicon_melt                            | Zheyong Fan   | NEP       | Melting point of silicon from two-phase method. |
| 09_Silicon_diffusion                       | Zheyong Fan   | NEP       | Diffusion coefficient of liquid silicon from VAC and MSD. |
| 10_Silicon_viscosity                       | Zheyong Fan   | NEP       | Viscosity of liquid silicon from Green-Kubo. |
| 11_NEP_potential_PbTe                      | Zheyong Fan   | NEP       | Train a NEP potential model for PbTe. |
| 12_NEP_dipole_QM7B                         | Nan Xu        | NEP       | Train a NEP dipole model for QM7B database. |
| 13_NEP_polarizability_QM7B                 | Nan Xu        | NEP       | Train a NEP polarizability model for QM7B database. |


## How to run the examples?

* First, compile the code by typing `make` in `src/`. You will get the executables `gpumd` and `nep` in `src/`.

* Then, go to the directory of an example and type one of the following commands:
  * `path/to/gpumd`
  * `path/to/nep`
  
* By default, the `nep` executable will use all the visible GPUs in the system. 
This is also the case for the `gpumd` executable when using a NEP model.
The visible GPU(s) can be set by the following command before running the code:
```
export CUDA_VISIBLE_DEVICES=[list of GPU IDs]
# examples:
export CUDA_VISIBLE_DEVICES=0 # only use GPU with ID 0
export CUDA_VISIBLE_DEVICES=1 # only use GPU with ID 1
export CUDA_VISIBLE_DEVICES=0,2 # use GPUs with ID 0 and ID 2
```
If you are using a job scheduling system such as `slurm`, you can set something as follows
```
#SBATCH --gres=gpu:v100:2 # using 2 V100 GPUs
```
We suggest use GPUs of the same type, otherwise a fast GPU will wait for a slower one.
The parallel efficiency of the `nep` executable is high (about 90%) unless you have a very small training data set or batch size.
The parallel efficiency of the 	`gpumd` executable depends on the number of atoms per GPU. Good parallel efficiency requires this number to be larger than about 1e5.

By default, the system is partitioned along the thickest direction, but one can overwrite this by specifying a partition direction in the following way:
```
potential YOUR_NEP_MODEL.txt   # use the default partition
potential YOUR_NEP_MODEL.txt x # force to partition along the x direction (the a direction for triclinic box)
potential YOUR_NEP_MODEL.txt y # force to partition along the y direction (the b direction for triclinic box)
potential YOUR_NEP_MODEL.txt z # force to partition along the z direction (the c direction for triclinic box)
```

## References

[1] Zheyong Fan, Yanzhou Wang, Penghua Ying, Keke Song, Junjie Wang, Yong Wang, Zezhu Zeng, Ke Xu, Eric Lindgren, J. Magnus Rahm, Alexander J. Gabourie, Jiahui Liu, Haikuan Dong, Jianyang Wu, Yue Chen, Zheng Zhong, Jian Sun, Paul Erhart, Yanjing Su, Tapio Ala-Nissila,
[GPUMD: A package for constructing accurate machine-learned potentials and performing highly efficient atomistic simulations](https://doi.org/10.1063/5.0106617), The Journal of Chemical Physics **157**, 114801 (2022).

