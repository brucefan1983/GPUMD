# `GPUMD`

## Manual
* We only maintain the online manual now: https://gpumd.zheyongfan.org

## Warnings in the beginning:

* Do not treat the code as a black box. Do as many tests as you can until you trust some part of it (the part you will use). The best way is to check the source code and make sure that it makes sense to you. Whenever you feel unsure about something regarding the code, you are welcome to ask questions in the mailing list.
  * You can use the following link to subscribe and unsubscribe the mailing list:
https://www.freelists.org/list/gpumd
  * To post a question, you can send an email to gpumd(at)freelists.org
  * Here is the archive (public): https://www.freelists.org/archive/gpumd/

* There are no functionalities for building simulation models. Users of `GPUMD` are supposed to be able to build simulation models by their own. 
  * One of the developers, Alexander J. Gabourie, has written some python scripts for pre-processing and post-processing data related to `GPUMD`. Here is the link: https://github.com/AlexGabourie/thermo
  * A graduate student, Ke Xu, is also publishing some python scripts for pre-processing and post-processing data related to `GPUMD`. Here is the link: https://github.com/Kick-H/For_gpumd

## What is `GPUMD`?

* `GPUMD` stands for Graphics Processing Units Molecular Dynamics. It is a new molecular dynamics (MD) code implemented fully on graphics processing units (GPUs). This code is highly efficient. For details, see Ref. [1].

* Force evaluation for many-body potentials has been significantly accelerated by using GPUs. Our efficient and flexible GPU implementation of the force evaluation for many-body potentials relies on a set of simple expressions for force, virial stress, and heat current derived in Ref. [2]. Detailed algorithms for efficient CUDA-implementation have been presented in Ref. [1]. We have implemented the following many-body potentials in `GPUMD`:
   * The EAM-type potential with some analytical forms
   * The general Tersoff potential with an arbitrary number of atom types
   * The REBO potential for Mo-S systems (2009)
   * The Stillinger-Weber (1985) potential with up to three atom types
   * The Vashishta (2007) potential
   
* Apart from being highly efficient, another unique feature of GPUMD is that it has useful utilities to study heat transport. The current version of `GPUMD` can calculate the following quantities related to heat transport:
   * It can calculate the phonon density of states (DOS) from the velocity autocorrelation (VAC).
   * It can calculate the equilibrium heat current autocorrelation (HAC), whose time integral gives the running thermal conductivity   according to the Green-Kubo relation. As stressed in Ref. [2], the heat current as implemented in LAMMPS does not apply to many-body  potentials and significantly underestimates the thermal conductivity in 2D materials described by many-body potentials. `GPUMD` also contains the thermal conductivity decomposition method as introduced in Ref. [3], which is useful for 2D materials.
   * It can calculate the thermal conductivity of a system of finite length or the thermal boundary resistance (Kapitza resistance) of an interface or similar structures using nonequilibrium MD (NEMD) methods. The spectral decomposition method as described in Ref. [3] has also been implemented.
   * It can calculate the thermal conductivity using the homogeneous nonequilibrium MD (HNEMD) method as detailed in Ref. [4].
     
* `GPUMD` was firstly used for heat transport simulations only but we are now making it more and more general. However, the functionalities in `GPUMD` are still very limited. We are working on implementing (1) more potential models, (2) more integrators (including external conditions), and (3) more measurements.

## Prerequisites

* You need to have a GPU card with compute capability no less than 3.5 and a `CUDA` toolkit no less than `CUDA` 9.0.
* Works for both linux and Windows operating systems. 
* We will try our best to keep `GPUMD` as a standalone code. So far, it does not depend on any other program other than the standard `C`, `C++`, and `CUDA` libraries.

## Compile GPUMD
* Go to the `src` directory and type `make`. When the compilation finishes, two executables, `gpumd` and `phoon`, will be generated in the `src` directory. 

## Run GPUMD
* Go to the directory where you can see `src`.
* Type `src/gpumd < examples/input_gpumd.txt` to run the examples in `examples/gpumd`.
* Type `src/phonon < examples/input_phonon.txt` to run the examples in `examples/phonon`.
* Please read the manual (https://gpumd.zheyongfan.org) to study the examples. These examples should get you started. 
  
## Authors:

* Zheyong Fan (Bohai University and Aalto University; Active developer)
  * brucenju(at)gmail.com
* Alexander J. Gabourie (Stanford University; Active developer)
  * gabourie(at)stanford.edu
* Ville Vierimaa (Aalto University; Not an active developer any more)
* Mikko Ervasti (Aalto University; Not an active developer any more)
* Ari Harju (Aalto University; Not an active developer any more)

## Citations

If you use `GPUMD` in your published work, we kindly ask you to cite the following paper which describes the central algorithms used in `GPUMD`:
* [1] Zheyong Fan, Wei Chen, Ville Vierimaa, and Ari Harju. Efficient molecular dynamics simulations with many-body potentials on graphics processing units. Computer Physics Communications **218**, 10 (2017). https://doi.org/10.1016/j.cpc.2017.05.003

If your work involves using heat current and virial stress formulas as implemented in `GPUMD`, the following paper can be cited:
* [2] Zheyong Fan, Luiz Felipe C. Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, and Ari Harju. Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations. Phys. Rev. B **92**, 094301, (2015). https://doi.org/10.1103/PhysRevB.92.094301

You can cite the following paper if you use `GPUMD` to study heat transport using the in-out decomposition for 2D materials and/or the spectral decomposition method as described in it:
* [3] Zheyong Fan, Luiz Felipe C. Pereira, Petri Hirvonen, Mikko M. Ervasti, Ken R. Elder, Davide Donadio, Tapio Ala-Nissila, and Ari Harju. Thermal conductivity decomposition in two-dimensional materials: Application to graphene. Phys. Rev. B **95**, 144309, (2017). https://doi.org/10.1103/PhysRevB.95.144309 

You can cite the following paper if you use `GPUMD` to study heat transport using the HNEMD method and the associated spectral decomposition method:
* [4] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium molecular dynamics method for heat transport and spectral decomposition with many-body potentials, Phys. Rev. B **99**, 064308 (2019). https://doi.org/10.1103/PhysRevB.99.064308
