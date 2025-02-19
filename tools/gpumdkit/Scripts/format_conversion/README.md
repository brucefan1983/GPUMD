### add_groups.py

---

This script adds group labels to structures based on specified elements.

#### Usage

```
python add_groups.py <filename> <Symbols>
```

- `<filename>`: The path to the input file (e.g., POSCAR).
- `<Symbols>`: Space-separated list of element symbols to group (e.g., Li Y Cl).

#### Example

```sh
python add_groups.py POSCAR Li Y Cl
```

This command will read the `POSCAR` file and add group labels for the elements `Li`, `Y`, and `Cl`. The output will be saved to a file named `model.xyz`.



### add_weight.py

---

This script adds weight labels to structures.

#### Usage

```
python add_weight.py <input_file> <output_file> <new_weight>
```

- `<inputfile>`: The path to the input file (e.g., train.xyz).
- `<outputfile>`: The path to the input file (e.g., train_weighted.xyz).
- `<new_weight>`: The `weight` you need to change.

#### Example

```sh
python add_weight.py train.xyz train_weighted.xyz 5
```

This command will read the `train.xyz` file and add `Weight=5` labels for all structures. The output will be saved to a file named `train_weighted.xyz`.



### exyz2pos.py

---

This script converts all frames in an `extxyz` file to `POSCAR` format.

#### Usage

```sh
python exyz2pos.py [extxyz_filename]
```

- `[extxyz_filename]`: (Optional) The path to the input `extxyz` file. If not specified, the default is `train.xyz`.

#### Example

```sh
python exyz2pos.py my_structures.xyz
```

This command will convert all frames in `my_structures.xyz` to `POSCAR` files.



### pos2exyz.py

---

This script converts a `POSCAR` file to `extxyz` format.

#### Usage

```
python pos2exyz.py <POSCAR_filename> <extxyz_filename>
```

- `<POSCAR_filename>`: The path to the input `POSCAR` file.
- `<extxyz_filename>`: The desired name for the output `extxyz` file.

#### Example

```
python pos2exyz.py POSCAR model.xyz
```

This command will read the `POSCAR` file and convert it to `model.xyz` in `extxyz` format.



### pos2lmp.py

---

This script converts a `POSCAR` file to `lammps-data` format.

#### Usage

```
python pos2lmp.py <poscar_file> <lammps_data_file>
```

- `<poscar_file>`: The path to the input `POSCAR` file.
- `<lammps_data_file>`: The desired name for the output `lammps-data` file.

#### Example

```
python pos2lmp.py POSCAR lammps.data
```

This command will read the `POSCAR` file and convert it to `lammps.data` in `lammps-data` format.



### split_single_xyz.py

---

This script splits an `extxyz` file into individual frames, each written to a separate file.

#### Usage

```
python split_single_xyz.py <extxyz_filename>
```

- `<extxyz_filename>`: The path to the input `extxyz` file.

#### Example

```sh
python split_single_xyz.py train.extxyz
```

This command will split all frames in `train.extxyz` into separate files named `model_${i}.xyz`, where `${i}` is the frame index.



### lmp2exyz.py

---

This script will convert the `lammps-dump` to `extxyz` format.

#### Usage

```
python lmp2exyz.py <dump_file> <element1> <element2> ...
```

- `<dump_file>`: The path to the input `lammps-dump` file.
- `<element>`: The order of the specified elements.

#### Example

```sh
python lmp2exyz.py dump.lammps Li Y Cl
```



### get_frame.py

---

This script will read the `extxyz` file and return the specified frame by index..

#### Usage

```
python get_frame.py <extxyz_file> <frame_index>
```

- `<extxyz_file>`: The path to the input `extxyz` file.
- `<frame_index>`: The index of the specified frame.

#### Example

```sh
python get_frame.py 1000
```

You will get the `frame_1000.xyz` file after perform the script.



---

Thank you for using `GPUMDkit`! If you have any questions or need further assistance, feel free to open an issue on our GitHub repository or contact Zihan YAN (yanzihan@westlake.edu.cn).
