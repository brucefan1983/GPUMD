# GPUMD

## What is GPUMD?

* GPUMD stands for Graphics Processing Units Molecular Dynamics. It is a new molecular dynamics (MD) code implemented fully on graphics processing units (GPUs). This code is highly efficient. For details, see Ref. [1].

* Force evaluation for many-body potentials has been significantly accelerated by using GPUs. Our efficient and flexible GPU implementation of the force evaluation for many-body potentials relies on a set of simple expressions for force, virial stress, and heat current derived in Ref. [2]. Detailed algorithms for efficient CUDA-implementation have been presented in Ref. [1]. We have implemented the following many-body potentials in GPUMD:
   * The EAM-type potential with some analytical forms
   * The Tersoff (1989) potential with single or double atom types
   * The Stillinger-Weber (1985) potential
   
* Apart from being highly efficient, another unique feature of GPUMD is that it has useful utilities to study heat transport. The current version of GPUMD can calculate the following quantities related to heat transport:
   * It can calculate the phonon density of states (DOS) from the velocity auto correlationfunction (VAC).
   * It can calculate the equilibrium heat current auto-correlation (HAC), whose time integral gives the running thermal conductivity  
     according to the Green-Kubo relation. As stressed in Ref. [2], the heat current as implemented in LAMMPS does not apply to many-body  
     potentials and significantly underestimates the thermal conductivity in 2D materials describ ed by many-body potentials. 
     GPUMD also contains the thermal conductivity decomposition method as introduced in Ref. [3], which is essential for 2D materials.
   * It can calculate the thermal conductivity of a system of finite length or the thermal boundary resistance (Kapitza resistance) of an 
     interface or similar structures using non-equilibrium MD (NEMD) methods. The spectral decompositions method as describ ed in Ref. [3] 
     has also been implemented.
     
* GPUMD was firstly used for heat transport simulations only but we are now making it more and more general.


# Citations

If you use GPUMD in your published work, we kindly ask you to cite the following paper which describes the central algorithms used in GPUMD:
* [1] Zheyong Fan, Wei Chen, Ville Vierimaa, and Ari Harju. Efficient molecular dynamics simulations with many-body potentials on graphics processing units. Computer Physics Communications, 218:10-16, 2017.

If your work involves using heat current and virial stress formulas as implemented in GPUMD, the following paper can be cited:
* [2] Zheyong Fan, Luiz Felipe C. Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, and Ari Harju. Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations. Phys. Rev. B, 92:094301, Sep 2015.

You can cite the following paper if you use GPUMD to study heat transport using the in-out decomposition for 2D materials and/or the spectral decomposition method as described in it:
* [3] Zheyong Fan, Luiz Felipe C. Pereira, Petri Hirvonen, Mikko M. Ervasti, Ken R. Elder, Davide Donadio, Tapio Ala-Nissila, and Ari Harju. Thermal conductivity decomposition in two-dimensional materials: Application to graphene. Phys. Rev. B, 95:144309, Apr 2017.
