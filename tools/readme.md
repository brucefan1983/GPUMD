# Tools related to GPUMD and NEP 

* Below is a list of the tools related to GPUMD and NEP (only the initial creator is listed).
* Hope you find them helpful, but you should use them with caution.
* If you have questions on a tool, you can try to contact the creator.




| Folder               | Creator      | Email                                       | Brief Description                                            |
| -------------------- | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| abacus2xyz           | Benrui Tang  | tang070205@proton.me   | Get `train.xyz` from `ABACUS` outputs.                       |
| add_groups           | Yuwen Zhang  | 984307703@qq.com | Generate grouping method(s) for `model.xyz`.                 |
| castep2exyz          | Yanzhou Wang | yanzhowang@gmail.com   | Get `train.xyz` from `CASTEP` outputs.                       |
| cp2k2xyz             | Zherui Chen  | chenzherui0124@foxmail.com | Get `train.xyz` from `CP2K` outputs or vice versa.           |
| deep2nep             | Ke Xu        | twtdq@qq.com                                     | Oudated?                                                     |
| doc_3.3.1            | Zheyong Fan  | brucenju@gmail.com | Documentation for some parts of GPUMD-v3.3.1.                |
| dp2xyz               | Ke Xu        | twtdq@qq.com        | Convert `DP` training data to `xyz` format.                  |
| exyz2pdb             | Zherui Chem  | chenzherui0124@foxmail.com  | Convert `exyz` to `pdb`. |
| for_coding           | Zheyong Fan  | brucenju@gmail.com     | Something useful for Zheyong Fan only.                       |
| get_max_rmse_xyz     | Ke Xu  | twtdq@qq.com | Identify structures with the largest errors.          |
| gmx2exyz             | Zherui Chen  | chenzherui0124@foxmail.com  | Convert the `trr` trajectory of `gmx` to the `exyz` trajectory. |
| gpumdkit             | Zihan Yan    | yanzihan@westlake.edu.cn            | A shell toolkit for GPUMD.                                   |
| md_tersoff           | Zheyong Fan  | brucenju@gmail.com  | Already in MD book; can be removed later.                    |
| mtp2nep              | Who? |                                                     | Outdated?                                                    |
| mtp2xyz              | Ke Xu | twtdq@qq.com       | Convert `MTP` training data to xyz format.                   |
| nep2xyz              | Ke Xu        | twtdq@qq.com                                        | Outdated?                                                    |
| pca_sampling         | Penghua Ying | hityingph@163.com | Farthest-point sampling based on `calorine`.                 |
| perturbed2poscar     | Who?         |                                                         | What?                                                        |
| rdf_adf              | Ke Xu        | twtdq@qq.com             | Calculate RDF and ADF using `OVITO`.                         |
| runner2xyz           | Ke Xu        | twtdq@qq.com    | Convert `RUNNER` training data to `xyz` format.             |
| select_xyz_frames    | Zherui Chen  | chenzherui0124@foxmail.com | Select frames from the `exyz`  file. |
| shift_energy_to_zero | Nan Xu       | tamas@zju.edu.cn | Shift the average energy of each species to zero for a dataset. |
| split_xyz            | Yong Wang    | yongw@princeton.edu | Some functionalities for training/test data.         |
| vasp2xyz             | Yanzhou Wang | yanzhowang@gmail.com     | Get `train.xyz` from `VASP` outputs.                         |
| vim                  | Ke Xu        | twtdq@qq.com                | Highlight GPUMD grammar in `vim`.                            |
| xyz2gro              | Who? |                             | Convert `xyz` file to `gro` file.                            |



## Python packages related to GPUMD and/or NEP:

| Package        | link                                           | comment                                                      |
| -------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| `calorine`     | https://gitlab.com/materials-modeling/calorine | `calorine` is a Python package for running and analyzing molecular dynamics (MD) simulations via GPUMD. It also provides functionality for constructing and sampling neuroevolution potential (NEP) models via GPUMD. |
| `GPUMD-Wizard` | https://github.com/Jonsnow-willow/GPUMD-Wizard | `GPUMD-Wizard` is a material structure processing software based on ASE (Atomic Simulation Environment) providing automation capabilities for calculating various properties of metals. Additionally, it aims to run and analyze molecular dynamics (MD) simulations using GPUMD. |
| `gpyumd`       | https://github.com/AlexGabourie/gpyumd         | `gpyumd` is a Python3 interface for GPUMD. It helps users generate input and process output files based on the details provided by the GPUMD documentation. It currently supports up to GPUMD-v3.3.1 and only the gpumd executable. |
| `mdapy`        | https://github.com/mushroomfire/mdapy          | The `mdapy` python library provides an array of powerful, flexible, and straightforward tools to analyze atomic trajectories generated from Molecular Dynamics (MD) simulations. |
| `pynep`        | https://github.com/bigd4/PyNEP                 | `PyNEP` is a python interface of the machine learning potential NEP used in GPUMD. |
| `somd`         | https://github.com/initqp/somd                 | `SOMD` is an ab-initio molecular dynamics (AIMD) package designed for the SIESTA DFT code. The SOMD code provides some common functionalities to perform standard Born-Oppenheimer molecular dynamics (BOMD) simulations, and contains a simple wrapper to the Neuroevolution Potential (NEP) package. The SOMD code may be used to automatically build NEPs by the mean of the active-learning methodology. |
| `NepTrainKit`  | https://github.com/aboys-cb/NepTrainKit        | `NepTrainKit` is a Python package for visualizing and manipulating training datasets for NEP. |
