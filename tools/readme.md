## Useful tools related to the `GPUMD` package

| folder      | author(s)                 | language | description                                               |
| ---------   | --------------------------| -------- | --------------------------------------------------------- |
| mtp2nep     | Junjie WANG               | Python   | Convert `MTP` training data to (old) `NEP` format |
| vim         | Ke Xu                     | vim      | highlight `GPUMD` related files in `vim`          |
| deep2nep    | Ke XU                     | Python   | Convert `DeePMD` training data to (old) `NEP` format | 
| nep2xyz     | Ke XU                     | Python   | Convert (old) `NEP` training data to extended XYZ file format |
| rdf_adf     | Ke XU                     | Python   | Calculate RDF and ADF usig `OVITO` |
| xyz2gro     | Nan XU                    | Python   | Convert extended XYZ file to `gro` file |
| vasp2xyz    | Yanzhou WANG, Yuwen ZHANG | Shell    | Create `NEP` training data from `VASP` output |
| castep2xyz  | Yanzhou WANG              | Shell    | Create `NEP` training data from `CASTEP` output |
| for_coding  | Zheyong FAN               | Matlab   | Used for developing `GPUMD` (the users can ignore this) |
| md_tersoff  | Zheyong FAN               | C++      | a standalone MD code implementing the calculations in Ref. [1] |
| doc_3.3.1   | Zheyong FAN               | PDF      | Some documentation for GPUMD-v3.3.1 |
| split_xyz   | Unkown                    | Python   | Select frames from an existing extended XYZ file. |

# References
* [1] Zheyong Fan, Luiz Felipe C Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, Ari Harju,
[Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations](https://doi.org/10.1103/PhysRevB.92.094301),
Physical Review B **92**, 094301 (2015).
