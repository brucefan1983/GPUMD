# `GPUMD`

## What is `GPUMD`?

* `GPUMD` stands for Graphics Processing Units Molecular Dynamics. It is a general-purpose molecular dynamics (MD) code fully implemented on graphics processing units (GPUs). 
* Force evaluation for many-body potentials has been significantly accelerated by using GPUs [1], thanks to a set of simple expressions for force, virial stress, and heat current derived in Refs. [2, 3]. 
* Apart from being highly efficient, another unique feature of GPUMD is that it has useful utilities to study heat transport [2, 3, 4, 5].
* It can run MD simulations with the machine-learning based force constant potential (FCP) [6].
* It can train the NEP machine-learning potential [7, 8] and run MD simulations with it.

## Prerequisites

* You need to have a GPU card with compute capability no less than 3.5 and a `CUDA` toolkit no older than `CUDA` 9.0.
* Works for both Linux (with GCC) and Windows (with MSVC) operating systems. 

## Compile GPUMD
* Go to the `src` directory and type `make`. When the compilation finishes, three executables, `gpumd`, `phonon`, and `nep`, will be generated in the `src` directory. 

## Run GPUMD
* See the `examples/readme.md` file.

## Manual
* We only maintain the online manual now: https://gpumd.zheyongfan.org

## Mailing list:
* You can use the following link to subscribe and unsubscribe the mailing list:
https://www.freelists.org/list/gpumd

* To post a question, you can send an email to gpumd(at)freelists.org

* Here is the archive (public): https://www.freelists.org/archive/gpumd/

## Python interface:

* One of the developers, Alexander J. Gabourie, has written a Python package for pre-processing and post-processing data related to `GPUMD`. Here is the link: https://github.com/AlexGabourie/thermo
  
## Authors:

* Zheyong Fan (Bohai University and Aalto University; Active developer)
  * brucenju(at)gmail.com
* Alexander J. Gabourie (Stanford University; Active developer)
  * gabourie(at)stanford.edu
* Ville Vierimaa (Aalto University; Not an active developer any more)
* Mikko Ervasti (Aalto University; Not an active developer any more)
* Ari Harju (Aalto University; Not an active developer any more)

## Citations

### Mandatory citation for any work used GPUMD:
* If you use `GPUMD` in your published work, we kindly ask you to cite the following paper which describes the central algorithms used in `GPUMD`:

[1] Zheyong Fan, Wei Chen, Ville Vierimaa, and Ari Harju. Efficient molecular dynamics simulations with many-body potentials on graphics processing units. Computer Physics Communications **218**, 10 (2017). https://doi.org/10.1016/j.cpc.2017.05.003

### Optional citation to the code repository:
* If you want to cite a link to the GPUMD code you can cite the current Github page: https://github.com/brucefan1983/GPUMD. 
* However, if the journal does not accept this citation, you can check the Zenodo page of GPUMD (https://zenodo.org/record/4037256#.X2jkqWj7SUk) and cite the version you used. Each version has a unique DOI, which is very suitable for citation. **Remember to change the author list to Zheyong Fan and Alexander J. Gabourie.**

### Other possible citations

* If your work involves using heat current and virial stress formulas as implemented in `GPUMD`, the following two papers can be cited:

[2] Zheyong Fan, Luiz Felipe C. Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, and Ari Harju. Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations. Phys. Rev. B **92**, 094301, (2015). https://doi.org/10.1103/PhysRevB.92.094301

[3] Alexander J. Gabourie, Zheyong Fan, Tapio Ala-Nissila, Eric Pop,
[Spectral Decomposition of Thermal Conductivity: Comparing Velocity Decomposition Methods in Homogeneous Molecular Dynamics Simulations](https://doi.org/10.1103/PhysRevB.103.205421),
Phys. Rev. B **103**, 205421 (2021).

* You can cite the following paper if you use `GPUMD` to study heat transport using the in-out decomposition for 2D materials and/or the spectral decomposition method as described in it:

[4] Zheyong Fan, Luiz Felipe C. Pereira, Petri Hirvonen, Mikko M. Ervasti, Ken R. Elder, Davide Donadio, Tapio Ala-Nissila, and Ari Harju. Thermal conductivity decomposition in two-dimensional materials: Application to graphene. Phys. Rev. B **95**, 144309, (2017). https://doi.org/10.1103/PhysRevB.95.144309 

* You can cite the following paper if you use `GPUMD` to study heat transport using the HNEMD method and the associated spectral decomposition method:

[5] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium molecular dynamics method for heat transport and spectral decomposition with many-body potentials, Phys. Rev. B **99**, 064308 (2019). https://doi.org/10.1103/PhysRevB.99.064308

* If you use the force constant potential (FCP), you can cite the following paper:

[6] Joakim Brorsson, Arsalan Hashemi, Zheyong Fan, Erik Fransson, Fredrik Eriksson, Tapio Ala-Nissila, Arkady V. Krasheninnikov, Hannu-Pekka Komsa, Paul Erhart, [Efficient calculation of the lattice thermal conductivity by atomistic simulations with ab-initio accuracy]( https://doi.org/10.1002/adts.202100217), Advanced Theory and Simulations **4**, 2100217 (2021). 

* If you train or use a NEP potential, you can cite the following papers:

[7] Zheyong Fan, Zezhu Zeng, Cunzhi Zhang, Yanzhou Wang, Keke Song, Haikuan Dong, Yue Chen, and Tapio Ala-Nissila, [Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport](https://doi.org/10.1103/PhysRevB.104.104309), Phys. Rev. B. **104**, 104309 (2021).

[8] Zheyong Fan, Improving the accuracy of the neuroevolution machine learning potentials for multi-component systems, https://arxiv.org/abs/2109.10643
