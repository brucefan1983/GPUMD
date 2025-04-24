# Lattice dynamics combined with neuroevolution potential (NEP) to compute lattice thermal conductivity of PbTe.
## _[Zezhu Zeng](https://scholar.google.com/citations?hl=en&user=3rOftLwAAAAJ&view_op=list_works&sortby=pubdate)_
Department of Mechnical Engineering, The University of Hong Kong, Hong Kong SAR, China.
Email: u3004964@connect.hku.hk


## Background

>This document provides a guide for computing the lattice thermal conductivity ($$\kappa$$) of thermoelectric material PbTe using lattice dynamics combined with the machine-learning-based neuroevolution potential (NEP). A key requirement for calculating $$\kappa$$ is the accurate determination of interatomic force constants (IFCs). We consider two approaches for extracting second- and third-order IFCs: the finite displacement method (FDM) and the temperature-dependent effective potential (TDEP) method [1]. FDM determines IFCs at $$0$$ K by introducing small atomic displacements and analyzing the resulting forces, while TDEP obtains effective IFCs at finite temperatures by accounting for atomic vibrations under realistic thermal conditions.

>Here we consistently employ a fitting-based approach to extract interatomic force constants (IFCs) for both the FDM and TDEP methods. Specifically, all atoms are displaced in each configuration, and the resulting displacement–force dataset is treated as training data for fitting. This strategy enables accurate determination of IFCs using significantly fewer configurations, thereby reducing computational cost.

>For the FDM approach, we use the HiPhive Python package to randomly generate Monte-Carlo-based configurations with small atomic displacements. The resulting displacement–force dataset is then used to fit the second- and third-order IFCs at 0 K. For the TDEP method, we perform classical molecular dynamics simulations at finite temperatures using the GPUMD package to generate the force–displacement dataset, which is subsequently used in HiPhive to fit temperature-dependent IFCs. 

## Required Software Packages
| Softwares | Links |
| ------ | ------ |
| HiPhive | [https://hiphive.materialsmodeling.org/][PlDb] |
| GPUMD | [https://github.com/brucefan1983/GPUMD][PlGh] |
| Phonopy | [https://phonopy.github.io/phonopy/][PlGd] |
| Phono3py | [https://phonopy.github.io/phono3py/][PlOd] |
| calorine | [https://calorine.materialsmodeling.org/][PlMe] |


## Finite displacement method 
### Dataset preparation

We begin by preparing two structural cells that define the equilibrium geometry for generating displaced configurations:

- **Relaxed primitive cell**: A fully relaxed unit cell representing the minimal repeating unit of the crystal.
- **Supercell**: A larger cell constructed by replicating the primitive cell, used to capture long-range atomic interactions for IFCs fitting.

We then generate the displaced atomic configurations and construct the corresponding displacement–force dataset using the [HiPhive](https://hiphive.materialsmodeling.org/) Python package. The following Python script outlines the procedure:

```python
from ase.io import read, write
from hiphive.structure_generation import generate_mc_rattled_structures
from calorine.calculators import CPUNEP
from hiphive.utilities import get_displacements
import numpy as np

# Parameters
n_structures = 100
rattle_std = 0.02
minimum_distance = 2.0

# Load equilibrium structures
prim = read('POSCAR', format='vasp')  # Relaxed primitive cell
atoms_ideal = read('SPOSCAR', format='vasp')  # Supercell used for IFCs extraction

# Set up NEP calculator
calc = CPUNEP('/path/to/nep.txt')

# Generate randomly rattled structures
structures_mc = generate_mc_rattled_structures(atoms_ideal, n_structures, rattle_std, minimum_distance)

# Build training dataset
training_structures = []
for atoms in structures_mc:
    atoms.calc = calc
    forces = atoms.get_forces()
    disps = get_displacements(atoms, atoms_ideal)

    atoms_tmp = atoms_ideal.copy()
    atoms_tmp.new_array('displacements', disps)
    atoms_tmp.new_array('forces', forces)
    training_structures.append(atoms_tmp)

# Write output
write('training_structures.extxyz', training_structures)
print('Total number of displaced configurations:', len(training_structures))
```
Note that the crucial parameter is the `rattle_std`, which is the standard deviation of the distribution of displacements from Monte Carlo sampling. To visualize the distribution of atomic displacements in the training dataset, we compute the magnitude of displacement vectors from the `displacements` array in each configuration and plot a histogram using the following Python code:

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Read all configurations from the extended XYZ file
structures = read('training_structures.extxyz', index=':')

# Collect all displacement magnitudes
displacement_magnitudes = []

for atoms in structures:
    displacements = atoms.arrays['displacements']  # shape: (n_atoms, 3)
    mags = np.linalg.norm(displacements, axis=1)
    displacement_magnitudes.extend(mags)
displacement_magnitudes = np.array(displacement_magnitudes)

# Plot histogram
plt.figure(figsize=(4, 3))
plt.hist(displacement_magnitudes, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Displacements (Å)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```
The displacement distribution (`rattle_std` = $$0.02$$) of the generated configurations from Monte Carlo sampling is shown below:
![Updated Image](https://raw.githubusercontent.com/ZengZezhu/figures_PbTe_tutorial/main/disp.png)


### Fitting Interatomic Force Constants (IFCs)

After generating the displacement–force dataset, we fit the second- and third-order interatomic force constants (IFCs) using the [HiPhive](https://hiphive.materialsmodeling.org/). The procedure involves the following steps:

- Reading displaced atomic configurations and associated forces.
- Defining a cluster space based on cutoff radii.
- Fitting the force constant parameters using regression.
- Constructing a force constant potential (FCP) model.
- Exporting the fitted IFCs in formats compatible with **Phono3py** or **ShengBTE**.

```python
from ase.io import read, write
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures
from trainstation import Optimizer
import h5py
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# Read reference structures
prim = read('POSCAR')
atoms_ideal = read('SPOSCAR')
structures = read('training_structures.extxyz', index=':')

# Define cluster space
cutoffs = [8.0, 6.0]
cs = ClusterSpace(prim, cutoffs)
print(cs)
cs.print_orbits()

# Add training data to structure container
sc = StructureContainer(cs)
for structure in structures:
    sc.add_structure(structure)
print(sc)

# Fit the model using Trainstation
opt = Optimizer(sc.get_fit_data())
opt.train()
print(opt)

# Build force constant potential
fcp = ForceConstantPotential(cs, opt.parameters)
fcp.write('fcc-pbte-highT.fcp')
print(fcp)

# Extract and export force constants
fcs = fcp.get_force_constants(atoms_ideal)
fcs.write_to_phonopy('FORCE_CONSTANTS_fdm_2nd', format='text')
fcs.write_to_shengBTE('FORCE_CONSTANTS_fdm_3rd', prim)
fcs.write_to_phonopy('fc2.hdf5')
fcs.write_to_phono3py('fc3.hdf5')
```
### Computing Lattice Thermal Conductivity ($$\kappa$$) with Phono3py

After obtaining the fitted second- and third-order interatomic force constants (IFCs), we compute the lattice thermal conductivity ($$\kappa$$) using the [Phono3py](https://atztogo.github.io/phono3py/) package.

To proceed, ensure the following files are placed in the same directory:

- The **primitive cell** file: `POSCAR`
- The **second-order force constants** file: `fc2.hdf5` 
- The **third-order force constants** file: `fc3.hdf5` 

One can then launch the Phono3py calculation with the following command:
```bash
phono3py --dim 5 5 5 --mesh 12 12 12 --br --fc2 --fc3
```

Explanation of Parameters:
- --dim 5 5 5: This defines the supercell dimensions used to generate the second- and third-order IFCs. A 5×5×5 supercell was created from the primitive cell to capture long-range anharmonic interactions.
- --mesh 12 12 12: This sets the q-point mesh for solving the phonon Boltzmann transport equation (BTE). A denser mesh improves the accuracy of $$\kappa$$. The mesh size should balance accuracy and computational cost.
- --br: Computes $$\kappa$$ using the relaxation time approximation.
- --fc2 and --fc3: Instruct Phono3py to read the second- and third-order IFCs from the specified files.

We emphasize that two parameters play a crucial role in determining the calculated lattice thermal conductivity:

1. `cutoffs`: These define the maximum interatomic distances considered when constructing interaction clusters. A cluster of lattice sites (i, j, k, ...) is included only if all pairwise distances between atoms in the cluster are smaller than the specified cutoff. Properly chosen cutoffs are essential to accurately capture the range of anharmonic interactions without overfitting or excessive computational cost.

2. `q-point mesh`: This mesh controls the resolution of the Brillouin zone integration in the solution of the phonon Boltzmann transport equation. A denser mesh usually provides more accurate predictions of $$\kappa$$. However, increasing the mesh size also increases the computational cost.

Careful convergence tests with respect to both parameters are essential to ensure reliable results. Below, we provide an example illustrating how the lattice thermal conductivity varies with different cutoff distances and q-point meshes. While this example may serve as a useful reference for guiding future calculations, we emphasize that convergence conditions can vary across different material systems. Nevertheless, the methodology demonstrated here remains broadly applicable and can be extended to other systems.

##### Convergence with Respect to Cutoff Distance

The first figure below demonstrates an example of how the choice of cutoff distance affects the calculated $$\kappa$$. Here, we fix the second-order cutoff to $$8.0$$ Å and vary the third-order cutoff from $$4.0$$ Å to $$6.0$$ Å. As shown, increasing the cutoff from $$4.0$$ Å to $$5.0$$ Å significantly alters the results, while the change from $$5.0$$ Å to $$6.0$$ Å leads to only minor differences, indicating convergence. However, it is important to note that simply increasing the cutoff does not always improve accuracy. The definition of interaction clusters is based on pairwise atomic distances between discrete neighbors in the crystal, not a continuous function. Therefore, careful convergence tests are necessary, especially for more complex materials where the range of interactions and crystal symmetry may demand different cutoff schemes.

![Cutoff Convergence](https://raw.githubusercontent.com/ZengZezhu/figures_PbTe_tutorial/main/diffcutoff.png)

##### Convergence with Respect to q-Point Mesh

The second figure presents the convergence behavior of $$\kappa$$ with respect to the q-point mesh used in solving the phonon Boltzmann transport equation. Denser meshes (e.g., $$12 \times 12 \times 12$$) offer better sampling of phonon scattering processes, especially at lower temperatures, leading to more accurate predictions. As seen, the results converge well between $$9 \times 9 \times 9$$ and $$12 \times 12 \times 12$$, while the $$6 \times 6 \times 6$$ mesh underestimates the thermal conductivity. This highlights the importance of performing convergence tests with respect to q-mesh density. We stress that more complex materials may require even denser meshes and additional testing to ensure reliable results.

![q-Mesh Convergence](https://raw.githubusercontent.com/ZengZezhu/figures_PbTe_tutorial/main/diffmesh.png)


## Temperature-dependent effective potential

For strongly anharmonic materials at high temperatures such as PbTe, temperature-dependent anharmonic renormalization to phonons and thermal conductivities could be crucial. The temperature-dependent effective potential (TDEP) method allows one to extract temperature-dependent force constants from molecular dynamics simulations, as described by Hellman *et al.* (2011) [1]. The core idea is to first perform molecular dynamics (MD) simulations to obtain temperature-dependent atomic trajectories. From these MD simulations, a displacement–force dataset is constructed and used to fit the renormalized second- and third-order IFCs, which are then employed to compute the phonon dispersion and the $$\kappa$$.

##### GPUMD Input for TDEP Dataset Generation
The following is an example `run.in` input file used in the [GPUMD](https://gpumd.org/) package to generate atomic trajectories for TDEP analysis. The simulation is carried out in two stages:

1. **NVT Equilibration**: A Nosé–Hoover chain thermostat (`nvt_nhc`) is used to equilibrate the system at $$300$$ K for  $$50,000$$ timesteps with a  $$1$$ fs time step.
2. **NVE Production Run**: The system then evolves under microcanonical ensemble (`nve`) for  $$10,000 $$ steps $$, and atomic positions and velocities are dumped for later analysis, resulting in a total of **100 configurations** for use in displacement–force dataset preparation.

```plaintext
potential /path/to/nep.txt
time_step       1

velocity        300

ensemble        nvt_nhc 300 300 100
dump_thermo     1000
run             50000

ensemble        nve
time_step       1
dump_thermo     1000
dump_exyz       100 0 1
run             10000
```

##### Fitting Temperature-Dependent IFCs from MD Trajectory

The script below performs the full workflow for fitting second- and third-order interatomic force constants (IFCs) using the TDEP method. The displacement–force dataset is extracted from `dump.xyz`, which contains 100 snapshots sampled from the final stage of an NVE molecular dynamics simulation at 300 K. These configurations reflect thermally equilibrated atomic motions and are used to construct a temperature-dependent force field.

```python
import glob
import numpy as np
from ase.io import read, write
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from trainstation import Optimizer
from hiphive.utilities import get_displacements
import matplotlib.pyplot as plt
import os

# Read reference structures
prim = read('POSCAR')
atoms_ideal = read('temp-300/model.xyz', format='extxyz')
supercell = read('temp-300/model.xyz', format='extxyz')

# Prepare training structures from MD snapshots
training_structures = []
k = 0
md_structs = read('temp-300/dump.xyz', format='extxyz', index=':')
for i, atoms in enumerate(md_structs):
    k += 1
    forces = atoms.get_forces()
    disps = get_displacements(atoms, atoms_ideal)
    assert np.linalg.norm(disps, axis=1).max() < 2.0

    atoms_tmp = atoms_ideal.copy()
    atoms_tmp.new_array('displacements', disps)
    atoms_tmp.new_array('forces', forces)
    training_structures.append(atoms_tmp)

write('training_structures.extxyz', training_structures)

print('Finished reading force–displacement dataset.')
print('Total number of configurations:', k)

# Set up cluster space and structure container
cutoff = [8, 6]
cs = ClusterSpace(atoms_ideal, cutoff, symprec=0.001)
print(cs)
cs.print_orbits()

sc = StructureContainer(cs)
for atoms in training_structures:
    sc.add_structure(atoms)
sc.write('structure_container.sc')
print(sc)

# Fit force constant potential
opt = Optimizer(sc.get_fit_data())
opt.train()
print(opt)

fcp = ForceConstantPotential(cs, opt.parameters)
fcp.write('pbte_tdep23nd_300K.fcp')
print(fcp)

# Extract and export force constants
fcs = fcp.get_force_constants(supercell)
fcs.write_to_phonopy('fc2.hdf5')
fcs.write_to_phonopy('FORCE_CONSTANTS_tdep300K_r2nd', format='text')
fcs.write_to_phono3py('fc3.hdf5')
```

With the renormalized IFCs at $$300$$ K obtained from the TDEP method, and the $$0$$ K IFCs from the finite displacement method (FDM), we can compare the resulting phonon dispersions. A noticeable hardening of low-frequency phonon modes is observed along the X–W–K path, indicating that temperature-induced frequency renormalization is significant at $$300$$ K and should not be neglected.

Additionally, using the Python code provided above, the temperature-dependent third-order IFCs can also be extracted. These IFCs can be used in conjunction with the Phono3py workflow discussed in the FDM section to compute the $$\kappa$$ at $$300$$ K, now incorporating anharmonic effects at finite temperature. As there are no major obstacles to carrying out this calculation, we leave its implementation to the reader. For those interested in further exploration, this may provide a straightforward extension of the TDEP approach [2].


![Updated Image](https://raw.githubusercontent.com/ZengZezhu/figures_PbTe_tutorial/main/pd_ab.png)


### References

[1] Hellman, O., Abrikosov, I. A., & Simak, S. I. (2013). *Temperature dependent effective potential method for accurate free energy calculations of solids*. Physical Review B, 88(14), 144301. https://doi.org/10.1103/PhysRevB.88.144301
[2] Zeng, Z., Chen, C., Zhang, C., Zhang, Q., & Chen, Y. (2022). *Critical phonon frequency renormalization and dual phonon coexistence in layered Ruddlesden–Popper inorganic perovskites*. Physical Review B, 105(18), 184303. https://doi.org/10.1103/PhysRevB.105.184303



