#### workflow_activate_learning_dev.sh

---

See this [Tutorial](../../Tutorials/Tutorials_workflow_activate_learning.md) for running the workflow script.



#### scf_batch_pretreatment.sh

---

This script automates the preprocessing of `POSCAR` or `extxyz` files for *self-consistent field* (`SCF`) calculations. The script includes the following steps:

1. Converts a `.xyz` file to `POSCAR` format using `GPUMDkit` if no `.vasp` files are found in the current directory.
2. Renames and organizes `.vasp` files into a `struct_fp` directory.
3. Creates individual directories for each `POSCAR` file, setting up symbolic links to the necessary `VASP` input files.
4. Generates a `presub.sh` script to automate running `VASP` `SCF` calculations.

#### Usage

1. Prepare the environment:

   Ensure all `.vasp` files or a single `.xyz` file are in the current directory.

2. Enter:

   ```bash
   bash scf_batch_pretreatment.sh
   ```

3. You will see the following prompt: 

   ```sh
    Starting SCF batch pretreatment...
    Found 8 .vasp files.
    >-------------------------------------------------<
    | This function calls the script in Scripts       |
    | Script: scf_batch_pretreatment.sh               |
    | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |
    >-------------------------------------------------<
   
    We recommend using the prefix to locate the structure.
    The folder name will be added to the second line of XYZ.
    config_type=<prefix>_<ID>
    ------------>>
    Please enter the prefix of directory (e.g. FAPBI3_iter01)
   ```

4. Enter the `prefix` of the folder name:

   ```sh
   FAPBI3_iter01
   ```

​		The script `scf_batch_pretreatment.sh` in the `Scripts` will be called to perform the pretreatment.

 5. You will see the following prompts:

    ```
     >-----------------------------------------------------<
     ATTENTION: Place POTCAR, KPOINTS and INCAR in 'fp' Dir.
     ATTENTION: Place POTCAR, KPOINTS and INCAR in 'fp' Dir.
     ATTENTION: Place POTCAR, KPOINTS and INCAR in 'fp' Dir.
     >-----------------------------------------------------<
    ```


You need to prepare the `POTCAR`, `KPOINTS`, and `INCAR` files and place them in a directory named `fp`.



#### md_sample_batch_pretreatment_gpumd.sh

---

This script automates the preprocessing of `POSCAR` or `extxyz` files for MD sampling using `GPUMD`. 

1. If `.vasp` files are found in the current directory, it will convert them to `extxyz` format to prepare the `model.xyz` file for `GPUMD`. If `.vasp` files are not found, the `.xyz` file will be read and all frames in it will be split into a individual sample.
2. Renames and organizes `.xyz` files into a `struct_md` directory.
3. Creates individual directories for each `model.xyz` file, setting up symbolic links to the necessary `GPUMD` input files.
4. Generates a `presub.sh` script to automate running MD simulations.

#### Usage

1. Prepare the environment:

   Ensure all `.vasp` files or a single `.xyz` file are in the current directory.

2. Enter:

   ```bash
   bash md_sample_batch_pretreatment_gpumd.sh
   ```

3. You will see the following prompt: 

   ```sh
    Starting MD sample batch pretreatment...
    No .vasp files found, but found one XYZ file.
    Converting it to model.xyz using GPUMDkit...
    All frames from "NEP-dataset.xyz" have been split into individual model files.
    20 model.xyz files were generated.
    >-------------------------------------------------<
    | This function calls the script in Scripts       |
    | Script: md_sample_batch_pretreatment.sh         |
    | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |
    >-------------------------------------------------<
   ```

4. You will see the following prompts:

   ````
   ```
   >-----------------------------------------------<
   ATTENTION: Place run.in and nep.txt in 'md' Dir. 
   ATTENTION: Place run.in and nep.txt in 'md' Dir. 
   ATTENTION: Place run.in and nep.txt in 'md' Dir. 
   >-----------------------------------------------<
   ```
   ````

You need to prepare the `run.in` and`nep.txt` files and place them in a directory named `md`.



---

Thank you for using `GPUMDkit`! If you have any questions or need further assistance, feel free to open an issue on our GitHub repository or contact Zihan YAN (yanzihan@westlake.edu.cn).
