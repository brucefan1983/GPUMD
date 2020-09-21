# `GPUMD`

## What is `GPUMD`?

* `GPUMD` stands for Graphics Processing Units Molecular Dynamics. It is a general-purpose molecular dynamics (MD) code fully implemented on graphics processing units (GPUs). 

* Force evaluation for many-body potentials has been significantly accelerated by using GPUs [1], thanks to a set of simple expressions for force, virial stress, and heat current derived in Ref. [2]. 
   
* Apart from being highly efficient, another unique feature of GPUMD is that it has useful utilities to study heat transport [3, 4].

## Prerequisites

* You need to have a GPU card with compute capability no less than 3.5 and a `CUDA` toolkit no older than `CUDA` 9.0.
* Works for both Linux (with GCC) and Windows (with MSVC) operating systems. 

## Compile GPUMD
* Go to the `src` directory and type `make`. When the compilation finishes, two executables, `gpumd` and `phonon`, will be generated in the `src` directory. 

## Run GPUMD
* Go to the directory where you can see `src`.
* Type `src/gpumd < examples/input_gpumd.txt` to run the examples in `examples/gpumd`.
* Type `src/phonon < examples/input_phonon.txt` to run the examples in `examples/phonon`.

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

### Optional citation for the force, heat current, and virial formulas:
* If your work involves using heat current and virial stress formulas as implemented in `GPUMD`, the following paper can be cited:

[2] Zheyong Fan, Luiz Felipe C. Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, and Ari Harju. Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations. Phys. Rev. B **92**, 094301, (2015). https://doi.org/10.1103/PhysRevB.92.094301

### Optional citation for the in-out decomposition for 2D materials:
* You can cite the following paper if you use `GPUMD` to study heat transport using the in-out decomposition for 2D materials and/or the spectral decomposition method as described in it:

[3] Zheyong Fan, Luiz Felipe C. Pereira, Petri Hirvonen, Mikko M. Ervasti, Ken R. Elder, Davide Donadio, Tapio Ala-Nissila, and Ari Harju. Thermal conductivity decomposition in two-dimensional materials: Application to graphene. Phys. Rev. B **95**, 144309, (2017). https://doi.org/10.1103/PhysRevB.95.144309 

### Optional citation for the HNEMD method and the related spectral decomposition method:
* You can cite the following paper if you use `GPUMD` to study heat transport using the HNEMD method and the associated spectral decomposition method:
[4] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium molecular dynamics method for heat transport and spectral decomposition with many-body potentials, Phys. Rev. B **99**, 064308 (2019). https://doi.org/10.1103/PhysRevB.99.064308
