<div align="left">
<img src="./logo/logo-main-arctic.png" width = "25%" />
</div>

# `GPUMD`

Copyright (2017) Zheyong Fan.
This is the GPUMD software package.
This software is distributed under the GNU General Public License (GPL) version 3.

## What is `GPUMD`?

* `GPUMD` stands for Graphics Processing Units Molecular Dynamics.
* `GPUMD` is a highly efficient general-purpose molecular dynamics (MD) package fully implemented on graphics processing units (GPUs).
* `GPUMD` enables training and using a class of machine-learned potentials (MLPs) called neuroevolution potentials (NEPs). See this [nep-data Gitlab repo](https://gitlab.com/brucefan1983/nep-data) for some of the published NEP potentials and the related training/testing data.

## Prerequisites

* You need to have a GPU card with compute capability no less than 3.5 and a `CUDA` toolkit no older than `CUDA` 9.0.
* Works for both Linux (with GCC) and Windows (with MSVC) operating systems. 

## Compile GPUMD
* Go to the `src` directory and type `make`.
* When the compilation finishes, two executables, `gpumd` and `nep`, will be generated in the `src` directory. 

## Run GPUMD
* Go to the directory of an example in the [examples directory](examples) and type one of the following commands:
  * `path/to/gpumd`
  * `path/to/nep`

## Tutorials
* We provide a [Colab Tutorial](https://colab.research.google.com/drive/1QnXAveZgzwut4Mvldsw-r2I0EWIsj1KA?usp=sharing) to show the workflow of the construction of a NEP model and its application in large-scale atomistic simulations for PbTe system. This will run entirely on Google's cloud virtual machine.
* We also provide many tutorials and examples in the [GPUMD-Tutorials repository](https://github.com/brucefan1983/GPUMD-Tutorials).

## Manual
* For users and developers:
  * Latest released version: https://gpumd.org/
  * Development version: https://gpumd.org/dev/
* For developers only:
  * [The developer guide](developers)

## Tools

Various tools for `GPUMD` and `NEP` can be found in [tools](./tools/readme.md).

## CPU version of NEP
There is a standalone C++ implementation of the neuroevolution potential (NEP) in the [NEP_CPU repository](https://github.com/brucefan1983/NEP_CPU), which serves as the engine for many Python packages and provides an interface to the [LAMMPS package](https://github.com/lammps/lammps).

## Citations

| Reference             | cite for what?                    |
| --------------------- | --------------------------------- |
| [Xu2025]                   | for any work that used `GPUMD`    |
| [Fan2017]                   | for historical citation   |
| [Fan2015]                 | virial and heat current formulation   |
| [Fan2017]                   | in-out decomposition and related spectral decomposition  |
| [Fan2019]                 | HNEMD and related spectral decomposition   |
| [Gabourie2021]                 | EMD and HNEMD based modal analyses   |
| [Brorsson2021]                   | force constant potential (FCP) |
| [Fan2021]                   | neuroevolution potential (NEP) and specifically NEP1 |
| [Fan2022JPCM]                   | NEP2 |
| [Fan2022JCP]                   | NEP3 |
| [Liu2023]                  | NEP + ZBL |
| [Ying2024]                  | NEP + D3 dispersion correction |
| [Shi2023]                  | MSST integrator for shock wave simulation |
| [Fan2024]                  | linear-scaling quantum transport |
| [Song2024]                  | NEP4 or UNEP-v1 (General-purpose machine-learned potential for 16 elemental metals and their alloys)|
| [Xu2024]                  | TNEP (tensorial NEP models of dipole and polarizability) |
| [Song2025]                  | MCMD (hybrid Monte Carlo and molecular dynamics simulations) |
| [Ying2025]                  | PIMD/TRPMD (path-integral molecular dynamics/thermostatted ring-polymer molecular dynamics) |
| [Pan2024]                  | NEMD and NPHug shock methods |
| [Jiang2025]                  | SW + ILP (hybrid Stillinger-Weber potential with anisotropic interlayer potential) |
| [Bu2025]                  | NEP + ILP (hybrid NEP with anisotropic interlayer potential) |
| [Liang2025]                  | NEP89 (Universal neuroevolution potential for inorganic and organic materials across 89 elements) |

## References

[Xu2025] Ke Xu, Hekai Bu, Shuning Pan, Eric Lindgren, Yongchao Wu, Yong Wang, Jiahui Liu, Keke Song, Bin Xu, Yifan Li, Tobias Hainer, Lucas Svensson, Julia Wiktor, Rui Zhao, Hongfu Huang, Cheng Qian, Shuo Zhang, Zezhu Zeng, Bohan Zhang, Benrui Tang, Yang Xiao, Zihan Yan, Jiuyang Shi, Zhixin Liang, Junjie Wang, Ting Liang, Shuo Cao, Yanzhou Wang, Penghua Ying, Nan Xu, Chengbing Chen, Yuwen Zhang, Zherui Chen, Xin Wu, Wenwu Jiang, Esme Berger, Yanlong Li, Shunda Chen, Alexander J. Gabourie, Haikuan Dong, Shiyun Xiong, Ning Wei, Yue Chen, Jianbin Xu, Feng Ding, Zhimei Sun, Tapio Ala-Nissila, Ari Harju, Jincheng Zheng, Pengfei Guan, Paul Erhart, Jian Sun, Wengen Ouyang, Yanjing Su, Zheyong Fan, [GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials]( https://doi.org/10.1002/mgea.70028), MGE Advances **3**, e70028 (2025).

[Fan2017] Zheyong Fan, Wei Chen, Ville Vierimaa, and Ari Harju. [Efficient molecular dynamics simulations with many-body potentials on graphics processing units](https://doi.org/10.1016/j.cpc.2017.05.003), Computer Physics Communications **218**, 10 (2017). 

[Fan2015] Zheyong Fan, Luiz Felipe C. Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, and Ari Harju. [Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations](https://doi.org/10.1103/PhysRevB.92.094301), Phys. Rev. B **92**, 094301, (2015). 

[Gabourie2021] Alexander J. Gabourie, Zheyong Fan, Tapio Ala-Nissila, Eric Pop,
[Spectral Decomposition of Thermal Conductivity: Comparing Velocity Decomposition Methods in Homogeneous Molecular Dynamics Simulations](https://doi.org/10.1103/PhysRevB.103.205421),
Phys. Rev. B **103**, 205421 (2021).

[Fan2017] Zheyong Fan, Luiz Felipe C. Pereira, Petri Hirvonen, Mikko M. Ervasti, Ken R. Elder, Davide Donadio, Tapio Ala-Nissila, and Ari Harju. [Thermal conductivity decomposition in two-dimensional materials: Application to graphene](https://doi.org/10.1103/PhysRevB.95.144309), Phys. Rev. B **95**, 144309, (2017).  

[Fan2019] Zheyong Fan, Haikuan Dong, Ari Harju, and Tapio Ala-Nissila, [Homogeneous nonequilibrium molecular dynamics method for heat transport and spectral decomposition with many-body potentials](https://doi.org/10.1103/PhysRevB.99.064308), Phys. Rev. B **99**, 064308 (2019). 

[Brorsson2021] Joakim Brorsson, Arsalan Hashemi, Zheyong Fan, Erik Fransson, Fredrik Eriksson, Tapio Ala-Nissila, Arkady V. Krasheninnikov, Hannu-Pekka Komsa, Paul Erhart, [Efficient calculation of the lattice thermal conductivity by atomistic simulations with ab-initio accuracy]( https://doi.org/10.1002/adts.202100217), Advanced Theory and Simulations **4**, 2100217 (2021). 

[Fan2021] Zheyong Fan, Zezhu Zeng, Cunzhi Zhang, Yanzhou Wang, Keke Song, Haikuan Dong, Yue Chen, and Tapio Ala-Nissila, [Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport](https://doi.org/10.1103/PhysRevB.104.104309), Phys. Rev. B. **104**, 104309 (2021).

[Fan2022JPCM] Zheyong Fan, [Improving the accuracy of the neuroevolution machine learning potentials for multi-component systems](https://iopscience.iop.org/article/10.1088/1361-648X/ac462b), Journal of Physics: Condensed Matter **34**, 125902 (2022).

[Fan2022JCP] Zheyong Fan, Yanzhou Wang, Penghua Ying, Keke Song, Junjie Wang, Yong Wang, Zezhu Zeng, Ke Xu, Eric Lindgren, J. Magnus Rahm, Alexander J. Gabourie, Jiahui Liu, Haikuan Dong, Jianyang Wu, Yue Chen, Zheng Zhong, Jian Sun, Paul Erhart, Yanjing Su, Tapio Ala-Nissila,
[GPUMD: A package for constructing accurate machine-learned potentials and performing highly efficient atomistic simulations](https://doi.org/10.1063/5.0106617), The Journal of Chemical Physics **157**, 114801 (2022).

[Liu2023] Jiahui Liu, Jesper Byggmästar, Zheyong Fan, Ping Qian, and Yanjing Su,
[Large-scale machine-learning molecular dynamics simulation of primary radiation damage in tungsten](https://doi.org/10.1103/PhysRevB.108.054312),
Phys. Rev. B **108**, 054312 (2023).

[Ying2024] Penghua Ying and Zheyong Fan,
[Combining the D3 dispersion correction with the neuroevolution machine-learned potential](https://doi.org/10.1088/1361-648X/ad1278),
Journal of Physics: Condensed Matter **36**, 125901 (2024).

[Shi2023] Jiuyang Shi, Zhixing Liang, Junjie Wang, Shuning Pan, Chi Ding, Yong Wang, Hui-Tian Wang, Dingyu Xing, and Jian Sun,
[Double-Shock Compression Pathways from Diamond to BC8 Carbon](https://doi.org/10.1103/PhysRevLett.131.146101),
Phys. Rev. Lett. **131**, 146101 (2023).

[Fan2024] Zheyong Fan, Yang Xiao, Yanzhou Wang, Penghua Ying, Shunda Chen, and Haikuan Dong,
[Combining linear-scaling quantum transport and machine-learning molecular dynamics to study thermal and electronic transports in complex materials](https://doi.org/10.1088/1361-648X/ad31c2),
Journal of Physics: Condensed Matter **36**, 245901 (2024).

[Song2024] Keke Song, Rui Zhao, Jiahui Liu, Yanzhou Wang, Eric Lindgren, Yong Wang, Shunda Chen, Ke Xu, Ting Liang, Penghua Ying, Nan Xu, Zhiqiang Zhao, Jiuyang Shi, Junjie Wang, Shuang Lyu, Zezhu Zeng, Shirong Liang, Haikuan Dong, Ligang Sun, Yue Chen, Zhuhua Zhang, Wanlin Guo, Ping Qian, Jian Sun, Paul Erhart, Tapio Ala-Nissila, Yanjing Su, Zheyong Fan,
[General-purpose machine-learned potential for 16 elemental metals and their alloys](https://doi.org/10.1038/s41467-024-54554-x),
Nature Communications **15**, 10208 (2024).

[Xu2024] Nan Xu, Petter Rosander, Christian Schäfer, Eric Lindgren, Nicklas Österbacka, Mandi Fang, Wei Chen, Yi He, Zheyong Fan, Paul Erhart,
[Tensorial properties via the neuroevolution potential framework: Fast simulation of infrared and Raman spectra](https://doi.org/10.1021/acs.jctc.3c01343),
J. Chem. Theory Comput. **20**, 3273 (2024).

[Song2025] Keke Song, Jiahui Liu, Shunda Chen, Zheyong Fan, Yanjing Su, Ping Qian, [Solute segregation in polycrystalline aluminum from hybrid Monte Carlo and molecular dynamics simulations with a unified neuroevolution potential](https://arxiv.org/abs/2404.13694),
arXiv:2404.13694 [cond-mat.mtrl-sci]

[Ying2025] Penghua Ying, Wenjiang Zhou, Lucas Svensson, Esmée Berger, Erik Fransson, Fredrik Eriksson, Ke Xu, Ting Liang, Jianbin Xu, Bai Song, Shunda Chen, Paul Erhart, Zheyong Fan, [Highly efficient path-integral molecular dynamics simulations with GPUMD using neuroevolution potentials: Case studies on thermal properties of materials](https://doi.org/10.1063/5.0241006),
J. Chem. Phys. **162**, 064109 (2025).

[Pan2024] Shuning Pan, Jiuyang Shi, Zhixin Liang, Cong Liu, Junjie Wang, Yong Wang, Hui-Tian Wang, Dingyu Xing, and Jian Sun, [Shock compression pathways to pyrite silica from machine learning simulations](https://doi.org/10.1103/PhysRevB.110.224101),
Phys. Rev. B **110**, 224101 (2024).

[Jiang2025] Wenwu Jiang, Ting Liang, Hekai Bu, Jianbin Xu, and Wengen Ouyang, [Moiré-driven interfacial thermal transport in twisted transition metal dichalcogenides](https://doi.org/10.1021/acsnano.4c12148),
ACS Nano **19**, 16287 (2025).

[Bu2025] Hekai Bu, Wenwu Jiang, Penghua Ying, Ting Liang, Zheyong Fan, and Wengen Ouyang, [Accurate modeling of LEGO-like vdW heterostructures: Integrating machine learned with anisotropic interlayer potentials](https://arxiv.org/abs/2504.12985),
arXiv:2504.12985 [physics.comp-ph].

[Liang2025] Ting Liang, Ke Xu, Eric Lindgren, Zherui Chen, Rui Zhao, Jiahui Liu, Esmée Berger, Benrui Tang, Bohan Zhang, Yanzhou Wang, Keke Song, Penghua Ying, Nan Xu, Haikuan Dong, Shunda Chen, Paul Erhart, Zheyong Fan, Tapio Ala-Nissila, Jianbin Xu, [NEP89: Universal neuroevolution potential for inorganic and organic materials across 89 elements](https://arxiv.org/abs/2504.21286), arXiv:2504.21286 [cond-mat.mtrl-sci].

[Huang2025] Hongfu Huang, Junhao Peng, Kaiqi Li, Jian Zhou, Zhimei Sun, [Efficient GPU-Accelerated Training of a Neuroevolution Potential with Analytical Gradients](http://arxiv.org/abs/2507.00528),
arXiv:2507.00528 [cond-mat.dis-nn; cond-mat.mtrl-sci; physics.comp-ph].