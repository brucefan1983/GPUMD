## Useful tools related to the `GPUMD` package

| folder      | description                                                  |
| ----------- | ------------------------------------------------------------ |
| rebo_mos2   | used for implementing the `rebo_mos2.cu` file in `src/force/` |
| md_tersoff  | a standalone MD code (written in C++) implementing the thermal conductivity calculations in Ref. [1] |
| create_xyz  | small programs to create some `xyz.in` files                 |
| vim         | used to highlight `GPUMD` related files in `vim`             |
| nep_related | tools related to the `nep` executable in `GPUMD`             |

# References
* [1] Zheyong Fan, Luiz Felipe C Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, Ari Harju,
[Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations](https://doi.org/10.1103/PhysRevB.92.094301),
Physical Review B **92**, 094301 (2015).
