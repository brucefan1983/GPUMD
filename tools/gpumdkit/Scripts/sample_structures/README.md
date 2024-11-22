#### sample_structures.py

---

This script samples structures from an `extxyz` file using either '`uniform`' or '`random`' sampling methods. The sampled structures are then written to the '`sampled_structures.xyz`' file.

`<extxyz_file>`: The path to the input extxyz file containing structures.

`<sampling_method>`: The sampling method to use. Can be '`uniform`' or '`random`'.

`<num_samples>`: The number of samples to extract from the input file.

#### Usage

```sh
python sample_structures.py <extxyz_file> <sampling_method> <num_samples>
```

#### Example

```sh
python sample_structures.py train.xyz uniform 10
```

This command will sample 10 structures uniformly from `train.xyz` and save them to `sampled_structures.xyz`.



#### get_min_dist.py

---

This script calculates the minimum atomic distance within a given system. The input is an `extxtz` file, and the script outputs the minimum distance between any two atoms in the structure.

#### Usage

```bash
python get_min_dist.py <extxyz_file>
```



#### perturb_structure.py

---

This function calls the `perturb_structure.py` script to generate the perturbed structures.

`<input.vasp>`: `filename.vasp`

`<pert_num>`: number of perturbed structures

`<cell_pert_fraction>`: A fraction determines how much (relatively) will cell deform

`<atom_pert_distance>`: A distance determines how far atoms will move (in angstrom).

`<atom_pert_style>`: `<uniform>`, `<normal>`, `<const>`

#### Usage

```sh
python perturb_structure.py <input.vasp> <pert_num> <cell_pert_fraction> <atom_pert_distance> <atom_pert_style>
```

#### Example

```sh
python perturb_structure.py FAPbI3.vasp 20 0.03 0.2 normal
```

This command will generate 20 perturbed structures.



---

Thank you for using `GPUMDkit`! If you have any questions or need further assistance, feel free to open an issue on our GitHub repository or contact Zihan YAN (yanzihan@westlake.edu.cn).
