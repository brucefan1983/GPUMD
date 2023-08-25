<div align="left">
<img src="./logo/logo-main-arctic.png" width = "25%" />
</div>

# `GPUMD`

## What is `GPUMD`?

* `GPUMD` stands for Graphics Processing Units Molecular Dynamics. It is a general-purpose molecular dynamics (MD) code fully implemented on graphics processing units (GPUs). 
* Force evaluation for many-body potentials has been significantly accelerated by using GPUs [1], thanks to a set of simple expressions for force, virial stress, and heat current derived in Refs. [2, 3]. 
* Apart from being highly efficient, another unique feature of GPUMD is that it has useful utilities to study heat transport [2, 3, 4, 5].
* It can run MD simulations with the machine-learning based force constant potential (FCP) [6].
* It can train the NEP machine-learning potential [7, 8, 9] and run MD simulations with it. See this [nep-data Gitlab repo](https://gitlab.com/brucefan1983/nep-data) for some of the published NEP potentials and the related training/testing data.

## Prerequisites

* You need to have a GPU card with compute capability no less than 3.5 and a `CUDA` toolkit no older than `CUDA` 9.0.
* Works for both Linux (with GCC) and Windows (with MSVC) operating systems. 

## Compile GPUMD
* Go to the `src` directory and type `make`. When the compilation finishes, two executables, `gpumd` and `nep`, will be generated in the `src` directory. 

## Run GPUMD
* Go to the directory of an example and type one of the following commands:
  * `path/to/gpumd`
  * `path/to/nep`

## Colab tutorial
* We provide a [Colab Tutorial](https://colab.research.google.com/drive/1QnXAveZgzwut4Mvldsw-r2I0EWIsj1KA?usp=sharing) to show the workflow of the construction of a NEP model and its application in large-scale atomistic simulations for PbTe system. This will run entirely on Google's cloud virtual machine. You can also check other offline tutorials in the examples.

## Manual
* https://gpumd.org/

## Mailing list:
* You can use the following link to subscribe and unsubscribe the mailing list:
https://www.freelists.org/list/gpumd

* To post a question, you can send an email to gpumd(at)freelists.org

* Here is the archive (public): https://www.freelists.org/archive/gpumd/

## Python interfaces:

| Package               | link                           |
| --------------------- | --------------------------------- |
| `calorine`            | https://gitlab.com/materials-modeling/calorine  |
| `gpyumd`              |https://github.com/AlexGabourie/gpyumd   |
| `pynep`               | https://github.com/bigd4/PyNEP   |
| `somd`                | https://github.com/initqp/somd  |

  
## Authors:

* Before the first release, GPUMD was developed by Zheyong Fan, with help from Ville Vierimaa (Previously Aalto University) and Mikko Ervasti (Previously Aalto University) and supervision from Ari Harju (Previously Aalto University).
* Below is the full list of contributors starting from the first release.

| Name                  | contact                           |
| --------------------- | --------------------------------- |
| Zheyong Fan           | https://github.com/brucefan1983   |
| Alexander J. Gabourie | https://github.com/AlexGabourie   |
| Ke Xu                 | https://github.com/Kick-H         |
| Ting Liang            | https://github.com/Tingliangstu   |
| Jiahui Liu            | https://github.com/Jonsnow-willow |
| Penghua Ying          | https://github.com/hityingph      |
| Real Name ?           | https://github.com/Lazemare       |
| Real Name ?           | https://github.com/initqp         |
| Yanzhou Wang          | https://github.com/Yanzhou-Wang   |
| Rui Zhao              | https://github.com/grtheaory      |
| Eric Lindgren         | https://github.com/elindgren      |
| Junjie Wang           | https://github.com/bigd4          |
| Yong Wang             | https://github.com/AmbroseWong    |
| Zhixin Liang          | https://github.com/liangzhixin-202169    |
| Paul Erhart           | https://materialsmodeling.org/ |
| Nan Xu                | https://github.com/tamaswells | 
| Shunda Chen           | https://github.com/shdchen |

## Citations

| Reference             | cite for what?                    |
| --------------------- | --------------------------------- |
| [1]                   | for any work that used `GPUMD`   |
| [2-3]                 | virial and heat current formulation   |
| [4]                   | in-out decomposition and related spectral decomposition  |
| [5]                   | HNEMD and related spectral decomposition   |
| [6]                   | force constant potential (FCP) |
| [7-9]                 | neuroevolution potential (NEP) |
| [10]                  | NEP + ZBL |

## References

[1] Zheyong Fan, Wei Chen, Ville Vierimaa, and Ari Harju. [Efficient molecular dynamics simulations with many-body potentials on graphics processing units](https://doi.org/10.1016/j.cpc.2017.05.003), Computer Physics Communications **218**, 10 (2017). 

[2] Zheyong Fan, Luiz Felipe C. Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, and Ari Harju. [Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations](https://doi.org/10.1103/PhysRevB.92.094301), Phys. Rev. B **92**, 094301, (2015). 

[3] Alexander J. Gabourie, Zheyong Fan, Tapio Ala-Nissila, Eric Pop,
[Spectral Decomposition of Thermal Conductivity: Comparing Velocity Decomposition Methods in Homogeneous Molecular Dynamics Simulations](https://doi.org/10.1103/PhysRevB.103.205421),
Phys. Rev. B **103**, 205421 (2021).

[4] Zheyong Fan, Luiz Felipe C. Pereira, Petri Hirvonen, Mikko M. Ervasti, Ken R. Elder, Davide Donadio, Tapio Ala-Nissila, and Ari Harju. [Thermal conductivity decomposition in two-dimensional materials: Application to graphene](https://doi.org/10.1103/PhysRevB.95.144309), Phys. Rev. B **95**, 144309, (2017).  

[5] Zheyong Fan, Haikuan Dong, Ari Harju, and Tapio Ala-Nissila, [Homogeneous nonequilibrium molecular dynamics method for heat transport and spectral decomposition with many-body potentials](https://doi.org/10.1103/PhysRevB.99.064308), Phys. Rev. B **99**, 064308 (2019). 

[6] Joakim Brorsson, Arsalan Hashemi, Zheyong Fan, Erik Fransson, Fredrik Eriksson, Tapio Ala-Nissila, Arkady V. Krasheninnikov, Hannu-Pekka Komsa, Paul Erhart, [Efficient calculation of the lattice thermal conductivity by atomistic simulations with ab-initio accuracy]( https://doi.org/10.1002/adts.202100217), Advanced Theory and Simulations **4**, 2100217 (2021). 

[7] Zheyong Fan, Zezhu Zeng, Cunzhi Zhang, Yanzhou Wang, Keke Song, Haikuan Dong, Yue Chen, and Tapio Ala-Nissila, [Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport](https://doi.org/10.1103/PhysRevB.104.104309), Phys. Rev. B. **104**, 104309 (2021).

[8] Zheyong Fan, [Improving the accuracy of the neuroevolution machine learning potentials for multi-component systems](https://iopscience.iop.org/article/10.1088/1361-648X/ac462b), Journal of Physics: Condensed Matter **34** 125902 (2022).

[9] Zheyong Fan, Yanzhou Wang, Penghua Ying, Keke Song, Junjie Wang, Yong Wang, Zezhu Zeng, Ke Xu, Eric Lindgren, J. Magnus Rahm, Alexander J. Gabourie, Jiahui Liu, Haikuan Dong, Jianyang Wu, Yue Chen, Zheng Zhong, Jian Sun, Paul Erhart, Yanjing Su, Tapio Ala-Nissila,
[GPUMD: A package for constructing accurate machine-learned potentials and performing highly efficient atomistic simulations](https://doi.org/10.1063/5.0106617), The Journal of Chemical Physics **157**, 114801 (2022).

[10] Jiahui Liu, Jesper Byggmästar, Zheyong Fan, Ping Qian, and Yanjing Su,
[Large-scale machine-learning molecular dynamics simulation of primary radiation damage in tungsten](https://doi.org/10.1103/PhysRevB.108.054312)
Phys. Rev. B **108**, 054312 (2023).
