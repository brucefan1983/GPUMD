# Function2 - Sample Structures

This script provides a menu-driven interface to perform various tasks related to structure sampling.

### Menu Options

```sh
------------>>
201) Sample structures from extxyz
202) Sample structures by pynep
203) Find the outliers in training set
204) Perturb structure
205) Developing ...
000) Return to the main menu
------------>>
Input the function number:
```

### Option 201: Sample Structures from extxyz

This option allows you to sample structures from an `extxyz` file using a specified method.

1. Select option `201` from the menu:

   ```sh
   201
   ```

2. You will see the following prompt:

   ```sh
   >-------------------------------------------------<
   | This function calls the script in Scripts       |
   | Script: sample_structures.py                    |
   | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |
   >-------------------------------------------------<
   Input <extxyz_file> <sampling_method> <num_samples>
   Sampling_method: 'uniform' or 'random'
   Examp: train.xyz uniform 50
   ------------>>
   ```

3. Enter the `extxyz` file name, sampling method, and number of samples:

   ```sh
   train.xyz uniform 50
   ```

4. The script `sample_structures.py` in the `Scripts` will be called to perform the sampling.

### Option 202: Sample structures by pynep

This function calls the `pynep_select_structs.py` in the `Scripts/sample_structures` to sampling the structures by `pynep`.

1. Select option `202` from the menu:

   ```sh
   202
   ```

2. You will see the following prompt:

   ```sh
   >-------------------------------------------------<
   | This function calls the script in Scripts       |
   | Script: pynep_select_structs.py                 |
   | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |
   >-------------------------------------------------<
   Input <sample.xyz> <train.xyz> <nep_model> <min_dist>
   Examp: dump.xyz train.xyz ./nep.txt 0.01
   ------------>>
   ```

   `<samle.xyz>`: extxyz file

   `<train.xyz>`: `train.xyz`

   `<nep_model>`: `nep.txt`

   `<min_dist>`: min_dist for pynep sampling

3. Enter the following parameters:

   ```sh
   dump.xyz train.xyz nep.txt 0.01
   ```



### Option 203: Find the outliers in training set

This function calls the `get_max_rmse_xyz.py` script to find outliers in a training set.

1. Select option `203` from the menu:

   ```sh
   203
   ```

2. You will see the following prompt:

   ```sh
   >-------------------------------------------------<
   | This function calls the script in GPUMD's tools |
   | Script: get_max_rmse_xyz.py                     |
   | Developer: Ke XU (kickhsu@gmail.com)            |
   >-------------------------------------------------<
   Input <extxyz_file> <*_train.out> <num_outliers>
   Examp: train.xyz energy_train.out 13 
   ------------>>
   ```

   `<extxyz_file>`: extxyz file

   `<*_train.out>`: `energy_train.out`/`force_train.out`/`virial_train.out``

   `<num_outliers>`: number of outliers

3. Enter the `extxyz` file name, <*_train.out>, and number of outliers:

   ```sh
   train.xyz energy_train.out 13 
   ```

4. The script `sample_structures.py` in the `Scripts` will be called to perform the sampling.

### Option 204: Perturb structure

This function calls the `perturb_structure.py` script to generate the perturbed structures.

1. Select option `204` from the menu:

   ```sh
   204
   ```

2. You will see the following prompt:

   ```sh
   >-------------------------------------------------<
   | This function calls the script in Scripts       |
   | Script: perturb_structure.py                    |
   | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |
   >-------------------------------------------------<
   Input <input.vasp> <pert_num> <cell_pert_fraction> <atom_pert_distance> <atom_pert_style>
   The default paramters for perturb are 20 0.03 0.2 uniform
   Examp: POSCAR 20 0.03 0.2 normal
   ------------>>
   ```

   `<input.vasp>`: filename.vasp

   `<pert_num>`: number of perturbed structures

   `<cell_pert_fraction>`: A fraction determines how much (relatively) will cell deform

   `<atom_pert_distance>`: A distance determines how far atoms will move (in angstrom).

   `<atom_pert_style>`: `<uniform>`, `<normal>`, `<const>`

3. Enter your parameters like:

   ```sh
   POSCAR 20 0.03 0.2 uniform
   ```

4. The script `perturb_structure.py` in the `Scripts` will be called to perform the perturbation.



---

Thank you for using `GPUMDkit`! If you have any questions or need further assistance, feel free to open an issue on our GitHub repository or contact Zihan YAN (yanzihan@westlake.edu.cn).

