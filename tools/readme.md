## Useful tools related to the `GPUMD` package

| folder      | author(s)                 | language | description                                               |
| ---------   | --------------------------| -------- | --------------------------------------------------------- |
| mtp2xyz     | Junjie WANG               | Python   | Convert `MTP` training data to (old) `XYZ` format |
| vim         | Ke Xu                     | vim      | highlight `GPUMD` related files in `vim`          |
| deep2xyz    | Ke XU                     | Python   | Convert `DeePMD` training data to (old) `XYZ` format | 
| nep2xyz     | Ke XU                     | Python   | Convert (old) `NEP` training data to extended XYZ file format |
| get_max_rmse_xyz     | Ke XU            | Python   | Select structures with large training errors. |
| rdf_adf     | Ke XU                     | Python   | Calculate RDF and ADF usig `OVITO` |
| runner2xyz  | Ke XU                     | Python   | Convert runner training data to `XYZ` format |
| xyz2gro     | Nan XU                    | Python   | Convert extended XYZ file to `gro` file |
| vasp2xyz    | Yanzhou WANG, Yuwen ZHANG | Shell    | Create `NEP` training data from `VASP` output |
| castep2xyz  | Yanzhou WANG              | Shell    | Create `NEP` training data from `CASTEP` output |
| for_coding  | Zheyong FAN               | Matlab   | Used for developing `GPUMD` (the users can ignore this) |
| md_tersoff  | Zheyong FAN               | C++      | a standalone MD code implementing the calculations in Ref. [1] |
| doc_3.3.1   | Zheyong FAN               | PDF      | Some documentation for GPUMD-v3.3.1 |
| cp2k2xyz    | Zherui Chen               | Python   | Create `NEP` training data from `CP2K` output. |
| split_xyz   | Unkown                    | Python   | Select frames from an existing extended XYZ file. |

# References
* [1] Zheyong Fan, Luiz Felipe C Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, Ari Harju,
[Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations](https://doi.org/10.1103/PhysRevB.92.094301),
Physical Review B **92**, 094301 (2015).
